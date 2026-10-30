"""Phase 0.1 regression suite: independent audit remediation.

Concurrency tests are deterministic where they can be (a barrier releases all
threads at once, and the assertion is an exact count) and stress tests where the
point is repetition. Every test runs offline.
"""

import json
import threading

import pytest

from agent_harness.blackboard import Blackboard, replace_authority
from agent_harness.budget import Budget
from agent_harness.config import Settings
from agent_harness.errors import (BudgetExceeded, ContractViolation, PermissionDenied,
                                  ProviderBadResponse, ProviderTimeout, ProviderUnavailable,
                                  ReservationError)
from agent_harness.orchestrator import Orchestrator
from agent_harness.permissions import WritePolicy
from agent_harness.plan import Plan, Step, parse_plan, validate_plan
from agent_harness.providers.base import Completion, Message, coerce_usage
from agent_harness.providers.fake import FakeProvider
from agent_harness.providers.openai_compat import OpenAICompatibleProvider
from agent_harness.registry import ModelProfile, ModelRegistry
from agent_harness.roster import Agent, Roster

PROFILE = ModelProfile("free/mini", "local", strengths=("cheap", "planning", "reasoning"))
PRICED = ModelProfile("paid/big", "vendor", strengths=("cheap", "planning"),
                      cost_per_1k_in=0.05, cost_per_1k_out=0.0)
REG = ModelRegistry([PROFILE])
REG_PRICED = ModelRegistry([PRICED])
ROSTER = Roster([Agent("researcher", "r", "free/mini", "read", ("findings",)),
                 Agent("writer", "w", "free/mini", "write", ("deliverable", "summary"))],
                registry=REG)
TRIAGE_OK = {"contains": "Decide the route", "text": '{"route": "direct", "agent": "writer"}'}
ANSWER = {"contains": "task", "text": "the answer"}


# --- D1: thread-safe budget ---------------------------------------------------
def test_concurrent_reservations_never_exceed_the_limit():
    quote = Budget(None).quote(PRICED, 1000, 0)      # $0.05
    affordable = 20                                   # $1.00 / $0.05
    for _round in range(20):
        budget = Budget(limit_usd=1.0)
        barrier = threading.Barrier(100)
        accepted, rejected = [], []
        lock = threading.Lock()

        def attempt():
            barrier.wait()
            try:
                budget.reserve(quote, "concurrent")
                with lock:
                    accepted.append(1)
            except BudgetExceeded:
                with lock:
                    rejected.append(1)

        threads = [threading.Thread(target=attempt) for _ in range(100)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(accepted) == affordable, "let {} through, {} affordable".format(
            len(accepted), affordable)
        assert len(rejected) == 100 - affordable
        assert budget.reserved_usd <= 1.0 + 1e-9
        assert budget.remaining_usd >= -1e-9


def test_concurrent_settlements_lose_no_updates():
    budget = Budget(limit_usd=100.0)
    quote = budget.quote(PRICED, 1000, 0)
    reservations = [budget.reserve(quote, "settle") for _ in range(200)]
    barrier = threading.Barrier(200)

    def settle(reservation):
        barrier.wait()
        budget.settle(reservation, 0.001)

    threads = [threading.Thread(target=settle, args=(r,)) for r in reservations]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert abs(budget.spent_usd - 0.2) < 1e-9, "lost a settlement update"
    assert budget.open_reservations == []


def test_double_settlement_racing_settles_exactly_once():
    budget = Budget(limit_usd=10.0)
    reservation = budget.reserve(budget.quote(PRICED, 1000, 0), "race")
    barrier = threading.Barrier(2)
    outcomes = []
    lock = threading.Lock()

    def settle():
        barrier.wait()
        try:
            budget.settle(reservation, 0.01)
            with lock:
                outcomes.append("settled")
        except ReservationError:
            with lock:
                outcomes.append("refused")

    threads = [threading.Thread(target=settle) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["refused", "settled"]
    assert budget.spent_usd == 0.01, "a duplicate settlement must not double-charge"


def test_repeated_and_unknown_reservation_operations_are_typed_failures():
    budget = Budget(limit_usd=1.0)
    reservation = budget.reserve(budget.quote(PRICED, 1000, 0), "x")
    budget.settle(reservation, 0.01)
    with pytest.raises(ReservationError):
        budget.settle(reservation, 0.01)
    with pytest.raises(ReservationError):
        budget.release(reservation)
    with pytest.raises(ReservationError):
        budget.settle("never-existed", 0.01)
    with pytest.raises(ReservationError):
        budget.release("never-existed")


def test_open_reservations_are_claimed_as_uncertain_not_free():
    budget = Budget(limit_usd=1.0)
    budget.reserve(budget.quote(PRICED, 1000, 0), "leaked")
    claimed = budget.claim_leaks()
    assert len(claimed) == 1 and claimed[0].uncertain
    assert budget.open_reservations == []
    assert budget.spent_usd == 0.0 and budget.uncertain_usd == 0.05


# --- F1: malformed usage ------------------------------------------------------
MSG = [Message("user", "hello there")]


@pytest.mark.parametrize("usage", [
    {"prompt_tokens": "many", "completion_tokens": 1},
    {"prompt_tokens": -1, "completion_tokens": 1},
    {"prompt_tokens": 1, "completion_tokens": True},
    {"prompt_tokens": float("nan"), "completion_tokens": 1},
    ["not", "an", "object"],
])
def test_malformed_usage_is_a_typed_provider_error(usage):
    provider = OpenAICompatibleProvider(
        "https://x.test/v1", "k", sleep=lambda s: None,
        transport=lambda u, h, b, t: (200, {"choices": [{"message": {"content": "ok"}}],
                                            "usage": usage}))
    with pytest.raises(ProviderBadResponse):
        provider.complete(MSG, model="m")


def test_absent_usage_is_estimated_and_flagged_rather_than_refused():
    provider = OpenAICompatibleProvider(
        "https://x.test/v1", "k", sleep=lambda s: None,
        transport=lambda u, h, b, t: (200, {"choices": [{"message": {"content": "hello there"}}]}))
    completion = provider.complete(MSG, model="m")
    assert completion.input_tokens > 0 and completion.output_tokens > 0
    assert completion.raw["usage_reported"] is False


def test_malformed_usage_mid_run_leaves_no_leak_and_a_terminal_status():
    class BadUsage(FakeProvider):
        def complete(self, messages, model, temperature=0.0, max_tokens=1024):
            raise ProviderBadResponse("usage.prompt_tokens was 'str'")

    orchestrator = Orchestrator(REG_PRICED, Roster(
        [Agent("writer", "w", "paid/big", "write", ("deliverable",))], registry=REG_PRICED),
        BadUsage(), Settings(max_cost_usd=1.0))
    with pytest.raises(ProviderBadResponse):
        orchestrator.run("task")
    assert orchestrator.budget.open_reservations == [], "reservation leaked"
    assert orchestrator.budget.uncertain_usd > 0, "a possibly-billed call must not be free"
    end = orchestrator.trace.of_kind("run_end")
    assert end and end[-1].fields["status"] == "failed"


# --- G1: invalid JSON on 200 --------------------------------------------------
class _RawResponse:
    status = 200

    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_invalid_json_on_http_200_is_a_bad_response(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: _RawResponse(b"<html>gateway timeout</html>"))
    provider = OpenAICompatibleProvider("https://x.test/v1", "k", sleep=lambda s: None)
    with pytest.raises(ProviderBadResponse) as exc:
        provider.complete(MSG, model="m")
    assert "not JSON" in str(exc.value) and "gateway timeout" in str(exc.value)
    assert "Bearer" not in str(exc.value), "diagnostics must not leak the credential"


def test_invalid_json_mid_run_reaches_the_normal_failure_path(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: _RawResponse(b"not json at all"))
    provider = OpenAICompatibleProvider("https://x.test/v1", "k", sleep=lambda s: None)
    orchestrator = Orchestrator(REG, ROSTER, provider, Settings(max_cost_usd=1.0))
    with pytest.raises(ProviderBadResponse):
        orchestrator.run("task")
    kinds = [e.kind for e in orchestrator.trace.events]
    assert kinds[-1] == "run_end" and "provider_call_uncertain" in kinds
    assert orchestrator.budget.open_reservations == []


# --- E1: uncertain expenditure ------------------------------------------------
def test_timeout_is_never_accounted_as_free():
    class TimeoutProvider(FakeProvider):
        def complete(self, messages, model, temperature=0.0, max_tokens=1024):
            raise ProviderTimeout("read timed out")

    orchestrator = Orchestrator(REG_PRICED, Roster(
        [Agent("writer", "w", "paid/big", "write", ("deliverable",))], registry=REG_PRICED),
        TimeoutProvider(), Settings(max_cost_usd=1.0))
    with pytest.raises(ProviderTimeout):
        orchestrator.run("task")
    assert orchestrator.budget.spent_usd == 0.0
    assert orchestrator.budget.uncertain_usd > 0, "a timed-out call may have been billed"
    uncertain = orchestrator.trace.of_kind("provider_call_uncertain")
    assert uncertain and uncertain[0].fields["uncertain_usd"] > 0
    end = orchestrator.trace.of_kind("run_end")[-1].fields
    assert end["status"] == "failed" and end["uncertain_usd"] > 0


def test_a_rejected_request_is_not_counted_as_uncertain():
    class AuthFailure(FakeProvider):
        def complete(self, messages, model, temperature=0.0, max_tokens=1024):
            from agent_harness.errors import ProviderAuthError
            raise ProviderAuthError("401")

    orchestrator = Orchestrator(REG_PRICED, Roster(
        [Agent("writer", "w", "paid/big", "write", ("deliverable",))], registry=REG_PRICED),
        AuthFailure(), Settings(max_cost_usd=1.0))
    with pytest.raises(Exception):
        orchestrator.run("task")
    assert orchestrator.budget.uncertain_usd == 0.0, "a 401 is not billed"
    assert orchestrator.trace.of_kind("provider_call_failed")[-1].fields["billing"] == "not_billed"


# --- C1: settlement overrun ---------------------------------------------------
def test_settlement_overrun_is_reported_and_blocks_further_calls():
    budget = Budget(limit_usd=0.06)
    quote = budget.quote(PRICED, 1000, 0)            # $0.05
    reservation = budget.reserve(quote, "underquoted")
    settlement = budget.settle(reservation, 0.07)    # provider charged more than quoted
    assert settlement.overrun_usd == pytest.approx(0.02)
    assert budget.overrun_usd == pytest.approx(0.02)
    assert budget.exhausted_reason, "an overrun past the limit must exhaust the budget"
    with pytest.raises(BudgetExceeded) as exc:
        budget.reserve(quote, "next call")
    assert "exhausted" in str(exc.value)


def test_overrun_appears_in_the_trace():
    orchestrator = Orchestrator(REG_PRICED, Roster(
        [Agent("writer", "w", "paid/big", "write", ("deliverable",))], registry=REG_PRICED),
        FakeProvider([TRIAGE_OK, ANSWER]), Settings(max_cost_usd=0.40))
    board = orchestrator._begin("task", {})
    quote = orchestrator.budget.quote(PRICED, 1000, 0)      # $0.05
    reservation = orchestrator.budget.reserve(quote, "probe")
    settlement = orchestrator.budget.settle(reservation, 0.45)   # $0.40 over the quote, past the limit
    orchestrator._note_settlement(settlement, "s1", "writer", "free/mini", "step", "call-1")
    event = orchestrator.trace.of_kind("budget_overrun")
    assert event and event[0].fields["overrun_usd"] == pytest.approx(0.40)
    assert orchestrator.trace.of_kind("budget_exhausted")
    assert board.read("task") == "task"


# --- H2 / I1: plan validation -------------------------------------------------
def test_duplicate_producers_are_rejected_before_execution():
    roster = Roster([Agent("researcher", "r", "free/mini", "x", ("findings",)),
                     Agent("researcher2", "r2", "free/mini", "y", ("findings",))], registry=REG)
    plan = Plan(steps=[Step(id="s1", agent="researcher", instruction="a", writes=("findings",)),
                       Step(id="s2", agent="researcher2", instruction="b", writes=("findings",))])
    with pytest.raises(ContractViolation) as exc:
        validate_plan(plan, roster, {"task"}, WritePolicy(roster), 8)
    assert "two producers" in str(exc.value)


def test_a_step_declaring_the_same_output_twice_is_rejected():
    plan = parse_plan(json.dumps({"steps": [{"id": "s1", "agent": "writer", "instruction": "x",
                                             "writes": ["deliverable", "deliverable"]}]}),
                      ROSTER, 8)
    with pytest.raises(ContractViolation):
        validate_plan(plan, ROSTER, {"task"}, WritePolicy(ROSTER), 8)


def test_self_dependency_is_rejected():
    plan = parse_plan(json.dumps({"steps": [{"id": "s1", "agent": "writer", "instruction": "x",
                                             "reads": ["deliverable"],
                                             "writes": ["deliverable"]}]}), ROSTER, 8)
    with pytest.raises(ContractViolation) as exc:
        validate_plan(plan, ROSTER, {"task"}, WritePolicy(ROSTER), 8)
    assert "own output" in str(exc.value)


def test_an_invalid_plan_costs_no_step_calls():
    script = [{"contains": "Decide the route", "text": '{"route": "decompose"}'},
              {"contains": "Produce the plan", "text": json.dumps({"steps": [
                  {"id": "s1", "agent": "writer", "instruction": "x",
                   "reads": ["deliverable"], "writes": ["deliverable"]}]})},
              {"contains": "Your step", "text": "should never run"}]
    provider = FakeProvider(script)
    orchestrator = Orchestrator(REG, ROSTER, provider, Settings())
    with pytest.raises(ContractViolation):
        orchestrator.run("task")
    assert len(provider.calls) == 2, "only triage and planning may spend; no step ran"
    assert orchestrator.board.keys() == ["task"]


# --- L2: blackboard bypass ----------------------------------------------------
def test_write_has_no_public_overwrite_bypass():
    board = Blackboard(policy=WritePolicy(ROSTER))
    board.write("deliverable", "first", written_by="writer", step_id="s1")
    with pytest.raises(TypeError):
        board.write("deliverable", "second", written_by="writer", step_id="s2",
                    allow_overwrite=True)
    assert board.read("deliverable") == "first"


def test_replace_requires_authority_and_a_reason_and_is_audited():
    board = Blackboard(policy=WritePolicy(ROSTER))
    board.write("deliverable", "first", written_by="writer", step_id="s1")
    with pytest.raises(PermissionDenied):
        board.replace("deliverable", "second", authority=object(), reason="tidy up")
    with pytest.raises(PermissionDenied):
        board.replace("deliverable", "second", authority=replace_authority, reason="  ")
    entry = board.replace("deliverable", "second", authority=replace_authority,
                          reason="corrected citation", step_id="s9")
    assert board.read("deliverable") == "second"
    assert entry.version == 2
    assert board.replacements[0]["reason"] == "corrected citation"
    assert len(board.replacements[0]["sha256"]) == 64


# --- retry attribution under concurrency --------------------------------------
class SharedProvider(FakeProvider):
    """Counts retries process-wide but reports attempts per call, as providers do."""
    attempt_ceiling = 2
    name = "shared"

    def __init__(self, script):
        super().__init__(script)
        self.retries_used = 0
        self._lock = threading.Lock()

    def complete(self, messages, model, temperature=0.0, max_tokens=1024):
        with self._lock:
            self.retries_used += 3        # another caller's retries, not ours
            noise = self.retries_used
        completion = super().complete(messages, model, temperature, max_tokens)
        completion.raw["attempts"] = (noise % 2) + 1   # 1 or 2
        completion.raw["provider"] = "fake"
        return completion


def test_retry_attribution_is_per_call_not_shared_state():
    provider = SharedProvider([TRIAGE_OK, ANSWER])
    orchestrator = Orchestrator(REG, ROSTER, provider, Settings())
    orchestrator.run("task")
    attempts = [e.fields["attempts"] for e in orchestrator.trace.of_kind("step_end")]
    assert attempts, "no calls recorded"
    assert all(a in (1, 2) for a in attempts), (
        "attempts must come from the call, not from a shared counter delta: {}".format(attempts))


def test_two_orchestrators_sharing_a_provider_keep_separate_attempt_counts():
    provider = SharedProvider([TRIAGE_OK, ANSWER])
    results = {}
    barrier = threading.Barrier(2)

    def run(name):
        orchestrator = Orchestrator(REG, ROSTER, provider, Settings())
        barrier.wait()
        orchestrator.run("task")
        results[name] = [e.fields["attempts"] for e in orchestrator.trace.of_kind("step_end")]

    threads = [threading.Thread(target=run, args=(n,)) for n in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert set(results) == {"a", "b"}
    for name, attempts in results.items():
        assert attempts and all(a in (1, 2) for a in attempts), (name, attempts)
        assert len(attempts) >= 2


# --- terminal evidence after unexpected exceptions ----------------------------
def test_unexpected_exception_still_yields_a_terminal_event_and_no_leak():
    class Exploding(FakeProvider):
        def complete(self, messages, model, temperature=0.0, max_tokens=1024):
            raise ValueError("something no one anticipated")

    orchestrator = Orchestrator(REG_PRICED, Roster(
        [Agent("writer", "w", "paid/big", "write", ("deliverable",))], registry=REG_PRICED),
        Exploding(), Settings(max_cost_usd=1.0))
    with pytest.raises(ValueError):
        orchestrator.run("task")
    end = orchestrator.trace.of_kind("run_end")
    assert end, "no terminal event for an unexpected failure"
    assert end[-1].fields["status"] == "failed"
    assert end[-1].fields["error"] in ("ValueError", "reservation_leak_shown_as_uncertain")
    assert end[-1].fields.get("unexpected") is True
    assert orchestrator.budget.open_reservations == []
    assert orchestrator.budget.uncertain_usd > 0

"""Phase 0 regression suite.

Each test reproduces a failure that existed in 0.2.0 or a rule this milestone
introduced. Every one runs offline against the scripted provider: no network, no
paid call, no live credential.
"""

import json
import socket

import pytest

from agent_harness.blackboard import ACTOR_SYSTEM, ACTOR_USER, Blackboard
from agent_harness.budget import Budget
from agent_harness.config import Settings
from agent_harness.errors import (BudgetExceeded, ContractViolation, DependencyError,
                                  OverwriteRejected, PermissionDenied, ProviderBadResponse,
                                  ProviderRateLimit, ProviderTimeout)
from agent_harness.orchestrator import Orchestrator
from agent_harness.permissions import WritePolicy
from agent_harness.plan import Step, parse_plan, validate_plan
from agent_harness.providers.base import Completion
from agent_harness.providers.fake import FakeProvider
from agent_harness.providers.openai_compat import OpenAICompatibleProvider
from agent_harness.registry import ModelProfile, ModelRegistry
from agent_harness.roster import Agent, Roster

FREE = ModelProfile("free/mini", "local", strengths=("cheap", "planning", "extraction"))
PAID = ModelProfile("paid/big", "vendor", strengths=("cheap", "planning", "reasoning"),
                    cost_per_1k_in=0.01, cost_per_1k_out=0.03)
UNPRICED = ModelProfile("metered/x", "nvidia", strengths=("cheap", "planning"),
                        pricing_known=False)
REG = ModelRegistry([FREE, ModelProfile("free/big", "local", strengths=("reasoning",))])
REG_PAID = ModelRegistry([PAID])
REG_UNPRICED = ModelRegistry([UNPRICED])
ROSTER = Roster([Agent("researcher", "r", "free/mini", "read it", ("findings",)),
                 Agent("analyst", "a", "free/big", "judge it", ("issues",)),
                 Agent("writer", "w", "free/mini", "write it", ("deliverable", "summary"))],
                registry=REG)
PAID_ROSTER = Roster([Agent("writer", "w", "paid/big", "write it", ("deliverable",))],
                     registry=REG_PAID)

TRIAGE_OK = {"contains": "Decide the route", "text": '{"route": "direct", "agent": "writer"}'}
TRIAGE_DECOMPOSE = {"contains": "Decide the route", "text": '{"route": "decompose"}'}
ANSWER = {"contains": "task", "text": "the answer"}


def one_step_plan(reads="", writes="deliverable", agent="writer", instruction="write it"):
    reads_json = json.dumps([reads]) if reads else "[]"
    return {"contains": "Produce the plan", "text": json.dumps(
        {"steps": [{"id": "s1", "agent": agent, "instruction": instruction,
                    "reads": json.loads(reads_json), "writes": [writes]}]})}


# 1 ---------------------------------------------------------------------------
def test_zero_budget_blocks_the_very_first_call():
    provider = FakeProvider([TRIAGE_OK, ANSWER])
    orchestrator = Orchestrator(REG_PAID, PAID_ROSTER, provider, Settings(max_cost_usd=0.0))
    with pytest.raises(BudgetExceeded) as exc:
        orchestrator.run("task")
    assert provider.calls == [], "no provider call may happen under a zero budget"
    assert orchestrator.trace.of_kind("step_end") == []
    assert orchestrator.trace.of_kind("run_end")[0].fields["status"] == "budget_exceeded"
    assert "triage" in str(exc.value)


# 2 ---------------------------------------------------------------------------
def test_zero_budget_blocks_the_planning_call():
    orchestrator = Orchestrator(REG_PAID, PAID_ROSTER, FakeProvider([TRIAGE_OK]),
                                Settings(max_cost_usd=0.05))
    board = orchestrator._begin("task", {})
    # Exhaust the allowance the way a real run does: a settled reservation, not by
    # reaching into budget state.
    quote = orchestrator.budget.quote(PAID, 10, 10)
    reservation = orchestrator.budget.reserve(quote, "triage")
    orchestrator.budget.settle(reservation, orchestrator.budget.limit_usd)
    orchestrator.provider.calls.clear()
    with pytest.raises(BudgetExceeded):
        orchestrator._plan("task", board)
    assert orchestrator.provider.calls == []


# 3 ---------------------------------------------------------------------------
class RetryingProvider(FakeProvider):
    """Two permitted attempts, so the quote must cover both."""
    attempt_ceiling = 2


def test_budget_reserves_for_permitted_retries_before_calling():
    single = Budget(limit_usd=None).quote_for_prompt(PAID, "x" * 400, 1024, attempts=1)
    orchestrator = Orchestrator(REG_PAID, PAID_ROSTER, RetryingProvider([ANSWER]),
                                Settings(max_cost_usd=round(single.usd * 1.5, 6)))
    with pytest.raises(BudgetExceeded) as exc:
        orchestrator.run("task")
    assert orchestrator.provider.calls == []
    assert exc.value.quote.attempts == 2, "the quote must include the retry allowance"


def test_attempts_are_visible_in_the_evidence():
    class FlakyOnce(FakeProvider):
        attempt_ceiling = 2
        name = "flaky"

        def complete(self, messages, model, temperature=0.0, max_tokens=1024):
            self.retries_used = getattr(self, "retries_used", 0)
            if not getattr(self, "_tripped", False):
                self._tripped = True
                self.retries_used += 1
                return super().complete(messages, model, temperature, max_tokens)
            return super().complete(messages, model, temperature, max_tokens)

    orchestrator = Orchestrator(REG, ROSTER, FlakyOnce([TRIAGE_OK, ANSWER]), Settings())
    orchestrator.run("task")
    attempts = {e.fields.get("step_id"): e.fields.get("attempts")
                for e in orchestrator.trace.of_kind("step_end")}
    assert attempts["direct"] >= 1


# 4 ---------------------------------------------------------------------------
def test_planner_cannot_grant_itself_a_protected_key():
    plan = parse_plan('{"steps": [{"id": "s1", "agent": "writer", "instruction": "x", '
                      '"writes": ["task"]}]}', ROSTER, 8)
    with pytest.raises(ContractViolation) as exc:
        validate_plan(plan, ROSTER, {"task"}, WritePolicy(ROSTER), 8)
    assert "reserved system key" in str(exc.value)


def test_a_plan_naming_a_protected_key_cannot_reach_execution():
    script = [TRIAGE_DECOMPOSE,
              {"contains": "Produce the plan", "text": '{"steps": [{"id": "s1", "agent": "writer",'
                                                       ' "instruction": "x", "writes": ["task"]}]}'},
              {"contains": "Produce the plan", "text": '{"steps": [{"id": "s1", "agent": "writer",'
                                                       ' "instruction": "x", "writes": ["task"]}]}'},
              {"contains": "x", "text": "hijacked"}]
    orchestrator = Orchestrator(REG, ROSTER, FakeProvider(script), Settings())
    with pytest.raises(ContractViolation):
        orchestrator.run("original task")
    assert orchestrator.board.read("task") == "original task"


# 5 ---------------------------------------------------------------------------
def test_agent_cannot_write_another_agents_output():
    policy = WritePolicy(ROSTER)
    board = Blackboard(policy=policy)
    with pytest.raises(PermissionDenied) as exc:
        board.write("issues", ["forged"], written_by="writer", step_id="s1")
    assert "not authorised" in str(exc.value)


def test_a_forged_write_is_audited_and_refused():
    orchestrator = Orchestrator(REG, ROSTER, FakeProvider([TRIAGE_OK, ANSWER]), Settings())
    board = orchestrator._begin("task", {})
    forged = Step(id="s1", agent="writer", instruction="x", writes=("issues",))
    with pytest.raises(PermissionDenied):
        orchestrator._write_outputs(forged, '{"issues": ["x"]}', board)
    rejected = orchestrator.trace.of_kind("write_rejected")
    assert rejected and rejected[0].fields["reason"] == "PermissionDenied"
    assert not board.has("issues")


# 6 ---------------------------------------------------------------------------
def test_user_input_cannot_overwrite_system_state():
    orchestrator = Orchestrator(REG, ROSTER, FakeProvider([TRIAGE_OK, ANSWER]), Settings())
    board = orchestrator._begin("task", {"codebase": {"a.py": "x"}})
    with pytest.raises(PermissionDenied):
        board.write("task", "user override", written_by="user", actor=ACTOR_USER)
    assert board.read("task") == "task"
    with pytest.raises(PermissionDenied):
        board.write("codebase", {"b.py": "y"}, written_by="user", actor=ACTOR_USER)
    assert board.read("codebase") == {"a.py": "x"}


def test_system_state_cannot_be_rewritten_after_seeding():
    board = Blackboard(policy=WritePolicy(ROSTER))
    board.write("task", "original", written_by="harness", actor=ACTOR_SYSTEM)
    board.close_seeding()
    with pytest.raises(PermissionDenied):
        board.write("task", "again", written_by="harness", actor=ACTOR_SYSTEM)


# 7 ---------------------------------------------------------------------------
def test_dependencies_are_satisfied_before_a_step_runs():
    steps = [Step(id="s2", agent="analyst", instruction="judge", reads=("findings",),
                  writes=("issues",)),
             Step(id="s1", agent="researcher", instruction="read", writes=("findings",)),
             Step(id="s3", agent="writer", instruction="write", reads=("issues",),
                  writes=("deliverable",))]
    from agent_harness.plan import Plan
    validated = validate_plan(Plan(steps=steps), ROSTER, {"task"}, WritePolicy(ROSTER), 8)
    assert [s.id for s in validated.steps] == ["s1", "s2", "s3"]
    assert validated.reordered is True


def test_reordering_is_recorded_rather_than_silent():
    script = [TRIAGE_DECOMPOSE,
              {"contains": "Produce the plan", "text": json.dumps({"steps": [
                  {"id": "s2", "agent": "analyst", "instruction": "judge",
                   "reads": ["findings"], "writes": ["issues"]},
                  {"id": "s1", "agent": "researcher", "instruction": "read",
                   "writes": ["findings"]},
                  {"id": "s3", "agent": "writer", "instruction": "write",
                   "reads": ["issues"], "writes": ["deliverable"]}]})},
              {"contains": "read", "text": '{"files": 1}'},
              {"contains": "judge", "text": '{"issues": ["one"]}'},
              {"contains": "write", "text": "done"}]
    orchestrator = Orchestrator(REG, ROSTER, FakeProvider(script), Settings())
    orchestrator.run("task")
    plan_event = orchestrator.trace.of_kind("plan")[0].fields
    assert plan_event["reordered"] is True
    assert plan_event["declared_order"] == ["s2", "s1", "s3"]
    order = [e.fields["step_id"] for e in orchestrator.trace.of_kind("step_start")]
    assert order == ["s1", "s2", "s3"]


# 8 ---------------------------------------------------------------------------
def test_circular_dependencies_are_rejected():
    from agent_harness.plan import Plan
    steps = [Step(id="a", agent="writer", instruction="x", reads=("issues",),
                  writes=("deliverable",)),
             Step(id="b", agent="analyst", instruction="y", reads=("deliverable",),
                  writes=("issues",))]
    with pytest.raises(ContractViolation) as exc:
        validate_plan(Plan(steps=steps), ROSTER, {"task"}, WritePolicy(ROSTER), 8)
    assert "circular dependency" in str(exc.value)


# 9 ---------------------------------------------------------------------------
def test_missing_required_input_stops_the_step_and_is_audited():
    script = [TRIAGE_DECOMPOSE, one_step_plan(reads="findings"), {"contains": "write it", "text": "x"}]
    orchestrator = Orchestrator(REG, ROSTER, FakeProvider(script), Settings())
    with pytest.raises(ContractViolation):
        orchestrator.run("task")
    assert orchestrator.trace.of_kind("run_end")[0].fields["status"] == "failed"


def test_a_reference_no_step_produces_is_rejected_before_execution():
    plan = parse_plan('{"steps": [{"id": "s1", "agent": "writer", "instruction": "x", '
                      '"reads": ["ghost"], "writes": ["deliverable"]}]}', ROSTER, 8)
    with pytest.raises(ContractViolation) as exc:
        validate_plan(plan, ROSTER, {"task"}, WritePolicy(ROSTER), 8)
    assert "no step produces" in str(exc.value)


# 10 --------------------------------------------------------------------------
def test_provider_timeout_is_normalised_and_retried_then_reported():
    def timeout_transport(url, headers, body, timeout):
        raise ProviderTimeout("read timed out after {}s".format(timeout))

    provider = OpenAICompatibleProvider("https://example.test/v1", "k", retries=1,
                                        transport=timeout_transport, sleep=lambda s: None)
    with pytest.raises(ProviderTimeout):
        provider.complete([], model="m")
    assert provider.retries_used == 1, "a timeout is retryable, within bounds"
    assert provider.calls == 2


# 11 --------------------------------------------------------------------------
def test_rate_limit_is_retried_then_surfaced_as_a_rate_limit():
    provider = OpenAICompatibleProvider(
        "https://example.test/v1", "k", retries=1, sleep=lambda s: None,
        transport=lambda url, headers, body, timeout: (429, {"error": {"message": "slow down"}}))
    with pytest.raises(ProviderRateLimit):
        provider.complete([], model="m")
    assert provider.calls == 2 and provider.retries_used == 1


# 12 --------------------------------------------------------------------------
def test_invalid_provider_response_is_a_distinct_failure():
    provider = OpenAICompatibleProvider(
        "https://example.test/v1", "k", sleep=lambda s: None,
        transport=lambda url, headers, body, timeout: (200, {"choices": []}))
    with pytest.raises(ProviderBadResponse):
        provider.complete([], model="m")


# 13 --------------------------------------------------------------------------
def test_multiple_output_keys_become_separate_artifacts():
    both = {"contains": "Produce the plan", "text": json.dumps(
        {"steps": [{"id": "s1", "agent": "writer", "instruction": "write it",
                    "writes": ["deliverable", "summary"]}]})}
    script = [TRIAGE_DECOMPOSE, both,
              {"contains": "write it", "text": json.dumps({"deliverable": "the answer",
                                                           "summary": "short"})}]
    orchestrator = Orchestrator(REG, ROSTER, FakeProvider(script), Settings())
    result = orchestrator.run("task")
    assert result.board["deliverable"] == "the answer"
    assert result.board["summary"] == "short", "each key gets its own value, not a copy"

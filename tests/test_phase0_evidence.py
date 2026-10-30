"""Phase 0 part two: recovery, evidence quality and secret handling."""

import json

import pytest

from agent_harness.blackboard import Blackboard
from agent_harness.config import Settings
from agent_harness.errors import ContractViolation, ProviderUnavailable
from agent_harness.orchestrator import Orchestrator
from agent_harness.permissions import WritePolicy
from agent_harness.plan import Plan, Step, validate_plan
from agent_harness.providers.base import Completion
from agent_harness.providers.fake import FakeProvider
from agent_harness.registry import ModelProfile, ModelRegistry
from agent_harness.trace import Trace
from agent_harness.roster import Agent, Roster

FREE = ModelProfile("free/mini", "local", strengths=("cheap", "planning"))
UNPRICED = ModelProfile("metered/x", "nvidia", strengths=("cheap", "planning"),
                        pricing_known=False)
REG = ModelRegistry([FREE])
REG_UNPRICED = ModelRegistry([UNPRICED])
ROSTER = Roster([Agent("researcher", "r", "free/mini", "read it", ("findings",)),
                 Agent("writer", "w", "free/mini", "write it", ("deliverable", "summary"))],
                registry=REG)
UNPRICED_ROSTER = Roster([Agent("writer", "w", "metered/x", "write it", ("deliverable",))],
                         registry=REG_UNPRICED)

TRIAGE_OK = {"contains": "Decide the route", "text": '{"route": "direct", "agent": "writer"}'}
ANSWER = {"contains": "task", "text": "the answer"}


def plan_dict(steps):
    return json.dumps({"steps": steps})


def test_a_step_declaring_two_outputs_must_return_both():
    plan = Plan(steps=[Step(id="s1", agent="writer", instruction="x",
                            writes=("deliverable", "summary"))])
    validated = validate_plan(plan, ROSTER, {"task"}, WritePolicy(ROSTER), 8)
    orchestrator = Orchestrator(REG, ROSTER, FakeProvider(), Settings())
    board = orchestrator._begin("task", {})
    with pytest.raises(ContractViolation) as exc:
        orchestrator._write_outputs(validated.steps[0], json.dumps({"deliverable": "only one"}),
                                    board)
    assert "declares outputs" in str(exc.value) or "missing" in str(exc.value)
    assert not board.has("deliverable")


def test_a_failed_run_still_yields_a_complete_trace():
    class Exploding(FakeProvider):
        name = "exploding"

        def complete(self, messages, model, temperature=0.0, max_tokens=1024):
            if "Decide the route" in messages[-1].content:
                return Completion(text='{"route": "direct", "agent": "writer"}', model=model)
            raise ProviderUnavailable("upstream 503")

    orchestrator = Orchestrator(REG, ROSTER, Exploding(), Settings())
    with pytest.raises(ProviderUnavailable):
        orchestrator.run("task")
    kinds = [e.kind for e in orchestrator.trace.events]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"
    for required in ("triage", "step_start", "provider_call_failed"):
        assert required in kinds
    end = orchestrator.trace.of_kind("run_end")[0].fields
    assert end["status"] == "failed" and end["error"] == "ProviderUnavailable"
    assert all(e.fields.get("run_id") == orchestrator.run_id
               for e in orchestrator.trace.events), "every event must carry the run id"


def test_recovery_after_a_retry_does_not_duplicate_or_widen_writes():
    script = [{"contains": "Decide the route", "text": '{"route": "decompose"}'},
              {"contains": "Produce the plan", "text": plan_dict([
                  {"id": "s1", "agent": "writer", "instruction": "write it",
                   "writes": ["deliverable"]}])},
              {"contains": "write it", "text": "", "finish_reason": "length", "once": True},
              {"contains": "write it", "text": "recovered answer"}]
    orchestrator = Orchestrator(REG, ROSTER, FakeProvider(script), Settings(max_output_tokens=8))
    result = orchestrator.run("task")
    assert result.deliverable == "recovered answer"
    artifacts = [a for a in orchestrator.trace.artifacts() if a["key"] == "deliverable"]
    assert len(artifacts) == 1, "a truncation retry must not write the artifact twice"
    assert artifacts[0]["version"] == 1
    assert set(result.board) == {"task", "deliverable"}


def test_secrets_never_reach_the_trace(tmp_path):
    secret = "nvapi-abcdefghijklmnop1234567890"
    provider = FakeProvider([TRIAGE_OK, {"contains": "task", "text": "answer " + secret}])
    # Supply the trace explicitly: a caller-owned evidence log must be preserved.
    orchestrator = Orchestrator(REG, ROSTER, provider,
                                Settings(trace_path=str(tmp_path / "trace.jsonl")),
                                trace=Trace(str(tmp_path / "trace.jsonl")))
    orchestrator.trace.record("probe", authorization="Bearer " + secret, note=secret,
                              api_key=secret)
    orchestrator.run("task")
    dump = json.dumps([e.as_dict() for e in orchestrator.trace.events])
    assert secret not in dump
    assert "Bearer " + secret not in dump
    assert "[redacted]" in dump


def test_estimated_and_actual_cost_are_recorded_separately():
    orchestrator = Orchestrator(REG, ROSTER, FakeProvider([TRIAGE_OK, ANSWER]), Settings())
    result = orchestrator.run("task")
    events = orchestrator.trace.of_kind("step_end")
    assert events and all("estimated_cost_usd" in e.fields and "cost_usd" in e.fields
                          for e in events)
    assert result.cost_usd == pytest.approx(orchestrator.trace.total_cost())
    assert orchestrator.trace.estimated_cost() >= orchestrator.trace.total_cost()


def test_unpriced_models_are_never_treated_as_free():
    blocked = FakeProvider([TRIAGE_OK, ANSWER])
    orchestrator = Orchestrator(REG_UNPRICED, UNPRICED_ROSTER, blocked,
                                Settings(max_cost_usd=0.0))
    from agent_harness.errors import BudgetExceeded
    with pytest.raises(BudgetExceeded):
        orchestrator.run("task")
    assert blocked.calls == [], "an unpriced model under a zero budget must not be called"

    allowed = Orchestrator(REG_UNPRICED, UNPRICED_ROSTER, FakeProvider([TRIAGE_OK, ANSWER]),
                           Settings(max_cost_usd=5.0))
    allowed.run("task")
    flags = [e.fields["conservative"] for e in allowed.trace.of_kind("step_end")]
    assert flags and all(flags), "unpriced calls must be flagged as conservative estimates"


def test_artifacts_carry_content_hashes_that_match_the_board():
    orchestrator = Orchestrator(REG, ROSTER, FakeProvider([TRIAGE_OK, ANSWER]), Settings())
    result = orchestrator.run("task")
    artifact = [a for a in result.artifacts if a["key"] == "deliverable"][0]
    assert len(artifact["sha256"]) == 64
    assert artifact["sha256"] == orchestrator.board.provenance()["deliverable"]["sha256"]


def test_overwrite_of_another_agents_artifact_is_refused():
    board = Blackboard(policy=WritePolicy(ROSTER))
    board.write("findings", ["first"], written_by="researcher", step_id="s1")
    with pytest.raises(Exception) as exc:
        board.write("findings", ["second"], written_by="writer", step_id="s2")
    assert type(exc.value).__name__ in ("PermissionDenied", "OverwriteRejected")
    assert board.read("findings") == ["first"]


def test_owning_agent_may_revise_its_own_artifact_and_history_is_kept():
    board = Blackboard(policy=WritePolicy(ROSTER))
    board.write("findings", ["v1"], written_by="researcher", step_id="s1")
    board.write("findings", ["v2"], written_by="researcher", step_id="s2")
    assert board.read("findings") == ["v2"]
    assert board.history("findings") == [None, ["v1"]]
    assert board.entries()[-1].version == 2


def test_the_trace_is_enough_to_reconstruct_the_run():
    orchestrator = Orchestrator(REG, ROSTER, FakeProvider([TRIAGE_OK, ANSWER]), Settings())
    result = orchestrator.run("task")
    fields = {e.kind: e.fields for e in orchestrator.trace.events}
    assert fields["run_start"]["policy"]["writer"] == ["deliverable", "summary"]
    assert fields["run_start"]["budget"]["limit_usd"] == orchestrator.settings.max_cost_usd
    assert fields["deliver"]["sha256"] == orchestrator.board.provenance()["deliverable"]["sha256"]
    assert fields["run_end"]["status"] == "delivered"
    step = orchestrator.trace.of_kind("step_end")[0].fields
    for key in ("call_id", "agent", "model", "input_tokens", "output_tokens", "cost_usd",
                "estimated_cost_usd", "duration_ms", "attempts", "finish_reason"):
        assert key in step, "missing evidence field: {}".format(key)

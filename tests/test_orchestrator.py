import pytest
from agent_harness.config import Settings
from agent_harness.errors import BudgetExceeded, ConfigError, ContractViolation, ProviderError
from agent_harness.orchestrator import Orchestrator
from agent_harness.providers.base import Completion
from agent_harness.providers.fake import FakeProvider
from agent_harness.registry import ModelProfile, ModelRegistry
from agent_harness.roster import Agent, Roster

REG = ModelRegistry([
    ModelProfile("local/mini", "local", strengths=("cheap", "planning", "extraction"), speed="fast"),
    ModelProfile("vendor/reason", "vendor", strengths=("reasoning",), speed="slow",
                 cost_per_1k_in=0.01, cost_per_1k_out=0.03),
])
ROSTER = Roster([
    Agent("researcher", "researcher", "local/mini", "read it", ("findings",)),
    Agent("analyst", "analyst", "vendor/reason", "judge it", ("issues",)),
    Agent("writer", "writer", "local/mini", "write it", ("deliverable",)),
], registry=REG)

DECOMPOSE = {"contains": "Decide the route",
             "text": '{"route": "decompose", "reason": "three jobs"}'}
PLAN = {"contains": "Produce the plan", "text": (
    '{"rationale": "read, judge, write", "steps": ['
    '{"id": "s1", "agent": "researcher", "instruction": "read the code", "writes": ["findings"]},'
    '{"id": "s2", "agent": "analyst", "instruction": "judge it", "reads": ["findings"], "writes": ["issues"]},'
    '{"id": "s3", "agent": "writer", "instruction": "write it up", "reads": ["issues"], "writes": ["deliverable"]}]}')}
S1 = {"contains": "read the code", "text": '{"files": 2}'}
S2 = {"contains": "judge it", "text": '{"issues": ["secret"]}'}
S3 = {"contains": "write it up", "text": "# Report\none issue"}


def run(script, **settings):
    orchestrator = Orchestrator(REG, ROSTER, FakeProvider(script), Settings(**settings) if settings else Settings())
    return orchestrator, orchestrator.run("Review this codebase")


def test_decompose_route_runs_every_step_and_delivers():
    _, result = run([DECOMPOSE, PLAN, S1, S2, S3])
    assert result.route == "decompose" and result.steps_run == 3
    assert sorted(result.board) == ["deliverable", "findings", "issues", "task"]
    assert result.deliverable.startswith("# Report")


def test_direct_route_bypasses_planning():
    script = [{"contains": "Decide the route", "text": '{"route": "direct", "agent": "writer"}'},
              {"contains": "Review", "text": "one shot answer"}]
    _, result = run(script)
    assert result.route == "direct" and result.steps_run == 1
    assert result.deliverable == "one shot answer"
    assert not any(e.kind == "plan" and e.fields.get("steps") == 3 for e in result.trace.events)


def test_cost_reported_equals_cost_in_trace():
    _, result = run([DECOMPOSE, PLAN, S1, S2, S3])
    assert result.cost_usd == pytest.approx(result.trace.total_cost())


def test_reasoning_step_is_routed_to_the_expensive_model_only():
    orchestrator, result = run([DECOMPOSE, PLAN, S1, S2, S3])
    models_used = {e.fields.get("step_id"): e.fields.get("model") for e in result.trace.of_kind("step_end")}
    assert models_used["s2"] == "vendor/reason"
    assert models_used["s1"] == "local/mini" and models_used["s3"] == "local/mini"


def test_triage_contract_failure_falls_back_and_says_so():
    _, result = run([{"contains": "Decide the route", "text": "I would rather chat about it"},
                     {"contains": "Review", "text": "fallback answer"}])
    assert result.triage_source == "fallback" and result.route == "direct"
    assert len(result.trace.of_kind("fallback")) == 1


def test_planner_contract_failure_falls_back_to_one_step():
    _, result = run([DECOMPOSE, {"contains": "Produce the plan", "text": "no json here"},
                     {"contains": "Review", "text": "single"}])
    assert result.steps_run == 1
    assert len(result.trace.of_kind("fallback")) == 1


def test_one_repair_attempt_happens_before_falling_back():
    orchestrator = Orchestrator(REG, ROSTER, FakeProvider([DECOMPOSE, PLAN, S1, S2, S3]), Settings())
    orchestrator.provider = FakeProvider([DECOMPOSE,
                                          {"contains": "not usable", "text": PLAN["text"]},
                                          {"contains": "Produce the plan", "text": "nope"},
                                          S1, S2, S3])
    result = orchestrator.run("Review this codebase")
    assert result.steps_run == 3, "the repaired plan should have been used"
    assert len(result.trace.of_kind("contract_violation")) == 1


def test_cost_budget_stops_the_run_with_its_trace():
    with pytest.raises(BudgetExceeded) as exc:
        run([DECOMPOSE, PLAN, S1, S2, S3], max_cost_usd=0.0001)
    assert exc.value.trace is not None and exc.value.trace.of_kind("step_end")


def test_plan_over_the_step_ceiling_is_rejected_at_planning_time():
    _, result = run([DECOMPOSE, PLAN, S1, S2, S3], max_steps=1)
    assert result.steps_run == 1, "an oversized plan must fall back, not run"
    assert len(result.trace.of_kind("fallback")) == 1


def test_wall_clock_budget_stops_the_run():
    ticks = iter(range(0, 10000, 50))          # 50s per observation
    orchestrator = Orchestrator(REG, ROSTER, FakeProvider([DECOMPOSE, PLAN, S1, S2, S3]),
                                Settings(max_seconds=10.0), clock=lambda: next(ticks))
    with pytest.raises(BudgetExceeded) as exc:
        orchestrator.run("Review this codebase")
    assert "wall clock" in str(exc.value)


def test_provider_failure_is_recorded_before_raising():
    class Exploding:
        name = "exploding"
        def complete(self, messages, model, temperature=0.0, max_tokens=1024):
            if "Decide the route" in messages[-1].content:
                return Completion(text='{"route": "direct", "agent": "writer"}', model=model)
            raise ProviderError("upstream 503")
    orchestrator = Orchestrator(REG, ROSTER, Exploding(), Settings())
    with pytest.raises(ProviderError):
        orchestrator.run("do it")
    assert orchestrator.trace.of_kind("step_error")


def test_empty_task_is_refused():
    with pytest.raises(ConfigError):
        Orchestrator(REG, ROSTER, FakeProvider(), Settings()).run("   ")


def test_inputs_land_on_the_board():
    orchestrator, _ = run([DECOMPOSE, PLAN, S1, S2, S3])
    result = orchestrator.run("task", inputs={"codebase": {"a.py": "x"}})
    assert result.board["codebase"] == {"a.py": "x"}


def test_trace_records_the_full_lifecycle_in_order():
    _, result = run([DECOMPOSE, PLAN, S1, S2, S3])
    kinds = [e.kind for e in result.trace.events]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"
    for expected in ("triage", "plan", "step_start", "step_end", "deliver"):
        assert expected in kinds
    assert result.trace.of_kind("step_end")[0].fields["estimated"] is True


def test_truncated_answer_buys_more_room_instead_of_failing_the_contract():
    script = [{"contains": "Decide the route", "text": "", "finish_reason": "length", "once": True},
              {"contains": "Decide the route", "text": '{"route": "direct", "agent": "writer"}'},
              {"contains": "Review", "text": "answer"}]
    orchestrator = Orchestrator(REG, ROSTER, FakeProvider(script), Settings(max_output_tokens=32))
    result = orchestrator.run("Review this codebase")
    assert result.route == "direct" and result.triage_source == "model"
    assert len(result.trace.of_kind("truncated")) == 1
    assert len(result.trace.of_kind("contract_violation")) == 0, "truncation is not a contract failure"


def test_triage_alone_routes_without_running_steps():
    orchestrator = Orchestrator(REG, ROSTER, FakeProvider([DECOMPOSE]), Settings())
    decision = orchestrator.triage("Review this codebase")
    assert decision.route == "decompose"
    assert len(orchestrator.trace.of_kind("step_end")) == 1, "only the triage call should have run"


def test_a_truncated_step_buys_more_room():
    script = [DECOMPOSE, {"contains": "Produce the plan", "text": (
        '{"steps": [{"id": "s1", "agent": "writer", "instruction": "write it up", "writes": ["deliverable"]}]}')},
        {"contains": "write it up", "text": "", "finish_reason": "length", "once": True},
        {"contains": "write it up", "text": "the real answer"}]
    orchestrator = Orchestrator(REG, ROSTER, FakeProvider(script), Settings(max_output_tokens=8))
    result = orchestrator.run("write something")
    assert result.deliverable == "the real answer"
    assert orchestrator.trace.of_kind("truncated")


def test_an_empty_step_write_fails_loudly_instead_of_shipping_a_blank_answer():
    script = [DECOMPOSE, {"contains": "Produce the plan", "text": (
        '{"steps": [{"id": "s1", "agent": "writer", "instruction": "write it up", "writes": ["deliverable"]}]}')},
        {"contains": "write it up", "text": "   "}]
    orchestrator = Orchestrator(REG, ROSTER, FakeProvider(script), Settings())
    with pytest.raises(ContractViolation) as exc:
        orchestrator.run("write something")
    assert "no content" in str(exc.value)
    assert orchestrator.trace.of_kind("empty_write")

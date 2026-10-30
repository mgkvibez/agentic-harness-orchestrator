import pytest
from agent_harness.errors import ContractViolation
from agent_harness.plan import extract_json, parse_plan, parse_triage
from agent_harness.registry import ModelProfile, ModelRegistry
from agent_harness.roster import Agent, Roster

REG = ModelRegistry([ModelProfile("m", "p", strengths=("cheap", "planning"))])
ROSTER = Roster([Agent("researcher", "researcher", "m", "read", ("findings",)),
                 Agent("writer", "writer", "m", "write", ("deliverable",))], registry=REG)


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_from_fenced_block():
    assert extract_json('here you go\n```json\n{"a": 2}\n```\nthanks') == {"a": 2}


def test_extract_json_from_prose():
    assert extract_json('Sure! {"a": 3} hope that helps') == {"a": 3}


def test_extract_json_without_json_raises():
    with pytest.raises(ContractViolation):
        extract_json("I would rather describe my feelings")


def test_triage_direct_needs_an_agent():
    with pytest.raises(ContractViolation) as exc:
        parse_triage('{"route": "direct"}', ROSTER)
    assert "name the agent" in str(exc.value)


def test_triage_rejects_unknown_agent():
    with pytest.raises(ContractViolation):
        parse_triage('{"route": "direct", "agent": "ghost"}', ROSTER)


def test_triage_rejects_unknown_route():
    with pytest.raises(ContractViolation):
        parse_triage('{"route": "vibes"}', ROSTER)


def test_triage_accepts_decompose():
    assert parse_triage('{"route": "decompose", "reason": "many jobs"}', ROSTER).route == "decompose"


def test_plan_parses_steps_and_coerces_string_keys():
    plan = parse_plan('{"steps": [{"agent": "writer", "instruction": "write", '
                      '"reads": "findings", "writes": "deliverable"}]}', ROSTER, 5)
    assert plan.steps[0].reads == ("findings",) and plan.steps[0].writes == ("deliverable",)


def test_plan_rejects_unknown_agent():
    with pytest.raises(ContractViolation) as exc:
        parse_plan('{"steps": [{"agent": "ghost", "instruction": "x", "writes": ["a"]}]}', ROSTER, 5)
    assert "unknown agent" in str(exc.value)


def test_plan_rejects_too_many_steps():
    steps = [{"id": "s{}".format(i), "agent": "writer", "instruction": "x", "writes": ["a"]} for i in range(4)]
    with pytest.raises(ContractViolation) as exc:
        parse_plan({"steps": steps} if False else __import__("json").dumps({"steps": steps}), ROSTER, 3)
    assert "ceiling" in str(exc.value)


def test_plan_rejects_step_without_output():
    with pytest.raises(ContractViolation) as exc:
        parse_plan('{"steps": [{"agent": "writer", "instruction": "x"}]}', ROSTER, 5)
    assert "writes nothing" in str(exc.value)


def test_plan_rejects_duplicate_step_ids():
    with pytest.raises(ContractViolation):
        parse_plan('{"steps": [{"id": "s1", "agent": "writer", "instruction": "x", "writes": ["a"]}, '
                   '{"id": "s1", "agent": "writer", "instruction": "y", "writes": ["b"]}]}', ROSTER, 5)

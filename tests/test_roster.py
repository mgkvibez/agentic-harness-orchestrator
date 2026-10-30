import json
import pytest
from agent_harness.errors import ConfigError
from agent_harness.registry import ModelProfile, ModelRegistry
from agent_harness.roster import Agent, Roster, load_roster

REG = ModelRegistry([ModelProfile("m", "p", strengths=("cheap",))])


def agent(name="researcher", writes=("findings",)):
    return Agent(name=name, role=name, model="m", system_prompt="do it", writes=writes)


def test_agent_needs_a_prompt():
    with pytest.raises(ConfigError) as exc:
        Agent(name="x", role="x", model="m", system_prompt="  ", writes=("a",))
    assert "system prompt" in str(exc.value)


def test_agent_needs_declared_writes():
    with pytest.raises(ConfigError) as exc:
        Agent(name="x", role="x", model="m", system_prompt="go", writes=())
    assert "blackboard key" in str(exc.value)


def test_roster_checks_model_references():
    with pytest.raises(ConfigError) as exc:
        Roster([Agent(name="x", role="x", model="ghost", system_prompt="go", writes=("a",))], registry=REG)
    assert "ghost" in str(exc.value)


def test_duplicate_agent_names_rejected():
    with pytest.raises(ConfigError):
        Roster([agent(), agent()], registry=REG)


def test_summary_lists_roles_and_keys():
    summary = Roster([agent()], registry=REG).summary()
    assert "researcher" in summary and "findings" in summary


def test_load_roster_from_file(tmp_path):
    path = tmp_path / "agents.json"
    path.write_text(json.dumps({"agents": [{"name": "w", "model": "m",
                                            "system_prompt": "write", "writes": ["deliverable"]}]}))
    roster = load_roster(str(path), registry=REG)
    assert roster.names() == ["w"]

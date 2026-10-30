import json
import pytest
from agent_harness.cli import main
from agent_harness.trace import Trace


def test_demo_runs_offline_and_explains_itself(capsys):
    assert main(["demo"]) == 0
    captured = capsys.readouterr()
    assert "behind the scenes" in captured.out and "Security review" in captured.out
    assert "no model was called" in captured.err


def test_demo_json_is_machine_readable(capsys):
    assert main(["demo", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["route"] in ("direct", "decompose") and payload["board_keys"]


def test_models_lists_profiles(capsys):
    assert main(["models"]) == 0
    assert "ollama/qwen2.5" in capsys.readouterr().out


def test_agents_lists_roster(capsys):
    assert main(["agents"]) == 0
    out = capsys.readouterr().out
    assert "analyst" in out and "deliverable" in out


def test_run_without_a_task_is_a_usage_error(capsys):
    assert main(["run"]) == 2


def test_no_subcommand_is_a_usage_error():
    assert main([]) == 2


def test_cost_budget_exits_with_its_own_code(capsys):
    assert main(["run", "Review this codebase", "--max-cost", "0"]) == 4
    assert "budget exceeded" in capsys.readouterr().err


def test_live_without_litellm_exits_5(capsys):
    try:
        import litellm  # noqa: F401
        pytest.skip("litellm is installed here, so the refusal path cannot be hit")
    except ImportError:
        pass
    assert main(["run", "Review this codebase", "--live"]) == 5
    assert "missing dependency" in capsys.readouterr().err


def test_trace_subcommand_summarises_a_file(tmp_path, capsys):
    trace = Trace()
    trace.record("step_end", step_id="s1", agent="analyst", model="m", cost_usd=0.001,
                 input_tokens=1, output_tokens=1, duration_ms=1)
    path = tmp_path / "t.jsonl"
    trace.to_jsonl(str(path))
    assert main(["trace", str(path)]) == 0
    assert "analyst" in capsys.readouterr().out


def test_bad_models_file_is_a_config_error(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    assert main(["models", "--models", str(path)]) == 2

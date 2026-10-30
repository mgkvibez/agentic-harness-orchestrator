import json
import pytest
from agent_harness.trace import Trace


def test_totals_count_spend_events_once():
    trace = Trace()
    trace.record("step_end", step_id="s1", cost_usd=0.002, input_tokens=10, output_tokens=5)
    trace.record("step_end", step_id="s2", cost_usd=0.001, input_tokens=20, output_tokens=7)
    # run_end also carries a cost field for humans; it must not be double-counted.
    trace.record("run_end", cost_usd=0.003)
    assert trace.total_cost() == pytest.approx(0.003)
    assert trace.total_tokens() == {"input": 30, "output": 12}


def test_summary_lists_steps_and_totals():
    trace = Trace()
    trace.record("step_end", step_id="s1", agent="analyst", model="big/reason",
                 cost_usd=0.001, input_tokens=1, output_tokens=1, duration_ms=5)
    summary = trace.summary()
    assert "analyst" in summary and "big/reason" in summary and "totals:" in summary


def test_of_kind_filters():
    trace = Trace()
    trace.record("run_start"); trace.record("fallback", phase="triage")
    assert len(trace.of_kind("fallback")) == 1


def test_jsonl_roundtrip(tmp_path):
    trace = Trace()
    trace.record("run_start", task="t")
    trace.record("step_end", step_id="s1", cost_usd=0.1)
    path = tmp_path / "trace.jsonl"
    trace.to_jsonl(str(path))
    loaded = Trace.load(str(path))
    assert len(loaded.events) == 2 and loaded.total_cost() == pytest.approx(0.1)


def test_live_trace_appends_to_disk(tmp_path):
    path = tmp_path / "live.jsonl"
    trace = Trace(str(path))
    trace.record("run_start", task="t")
    assert json.loads(path.read_text().strip())["kind"] == "run_start"

import json
import pytest
from agent_harness.blackboard import Blackboard
from agent_harness.errors import HarnessError


def test_write_read_and_require():
    board = Blackboard()
    board.write("findings", [1, 2], written_by="researcher", step_id="s1")
    assert board.read("findings") == [1, 2]
    assert board.require("findings") == [1, 2]
    with pytest.raises(HarnessError):
        board.require("missing")


def test_versions_and_history_are_recorded():
    board = Blackboard()
    board.write("k", "v1", written_by="a")
    board.write("k", "v2", written_by="b")
    assert [e.version for e in board.entries()] == [1, 2]
    assert board.history("k") == [None, "v1"]


def test_slice_only_includes_existing_keys():
    board = Blackboard()
    board.write("a", 1)
    assert board.slice(["a", "b"]) == {"a": 1}


def test_per_value_limit_enforced():
    board = Blackboard(char_limit=50)
    with pytest.raises(HarnessError) as exc:
        board.write("big", "x" * 80)
    assert "over the" in str(exc.value)


def test_total_board_limit_enforced():
    board = Blackboard(char_limit=40)
    board.write("a", "x" * 30)
    with pytest.raises(HarnessError) as exc:
        board.write("b", "y" * 30)
    assert "would exceed" in str(exc.value)


def test_rewriting_a_key_does_not_count_twice_against_the_limit():
    board = Blackboard(char_limit=40)
    board.write("a", "x" * 30)
    board.write("a", "y" * 30)          # same key, same size: allowed
    assert board.total_chars() <= 40


def test_non_serialisable_value_rejected():
    board = Blackboard()
    with pytest.raises(HarnessError):
        board.write("fn", lambda x: x)


def test_save_writes_valid_json(tmp_path):
    board = Blackboard()
    board.write("a", {"x": 1}, written_by="t")
    path = tmp_path / "board.json"
    board.save(str(path))
    payload = json.loads(path.read_text())
    assert payload["values"]["a"] == {"x": 1} and payload["entries"][0]["key"] == "a"

import json
import pytest
from agent_harness.errors import ConfigError
from agent_harness.registry import ModelProfile, ModelRegistry, load_profiles

FREE = ModelProfile("local/tiny", "local", strengths=("cheap", "extraction"), speed="fast")
CODING = ModelProfile("big/code", "vendor", strengths=("coding",), speed="balanced",
                      cost_per_1k_in=0.003, cost_per_1k_out=0.015)
REASON = ModelProfile("big/reason", "vendor", strengths=("reasoning", "coding"), speed="slow",
                      cost_per_1k_in=0.01, cost_per_1k_out=0.03)


def test_unknown_strength_is_rejected():
    with pytest.raises(ConfigError) as exc:
        ModelProfile("x", "p", strengths=("telepathy",))
    assert "unknown strength" in str(exc.value)


def test_bad_speed_is_rejected():
    with pytest.raises(ConfigError):
        ModelProfile("x", "p", speed="ludicrous")


def test_negative_cost_is_rejected():
    with pytest.raises(ConfigError):
        ModelProfile("x", "p", cost_per_1k_in=-1)


def test_cost_math():
    assert CODING.cost(1000, 1000) == pytest.approx(0.018)


def test_cheapest_with_prefers_free_then_cost():
    registry = ModelRegistry([REASON, CODING, FREE])
    assert registry.cheapest_with("coding").id == "big/code"
    assert registry.cheapest_with("cheap").id == "local/tiny"


def test_cheapest_with_can_exclude_models():
    registry = ModelRegistry([REASON, CODING])
    assert registry.cheapest_with("coding", exclude=["big/code"]).id == "big/reason"


def test_missing_strength_names_the_problem():
    with pytest.raises(ConfigError) as exc:
        ModelRegistry([FREE]).cheapest_with("reasoning")
    assert "reasoning" in str(exc.value)


def test_unknown_model_lookup_lists_known_ids():
    with pytest.raises(ConfigError) as exc:
        ModelRegistry([FREE]).get("nope")
    assert "local/tiny" in str(exc.value)


def test_duplicate_ids_rejected():
    with pytest.raises(ConfigError):
        ModelRegistry([FREE, FREE])


def test_empty_registry_rejected():
    with pytest.raises(ConfigError):
        ModelRegistry([])


def test_load_profiles_from_file(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"models": [{"id": "a/b", "strengths": ["reasoning"],
                                            "cost_per_1k_in": 0.5}]}))
    profiles = load_profiles(str(path))
    assert profiles[0].id == "a/b" and profiles[0].cost_per_1k_in == 0.5


def test_load_profiles_rejects_entry_without_id(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps([{"strengths": ["reasoning"]}]))
    with pytest.raises(ConfigError):
        load_profiles(str(path))


def test_rank_breaks_ties_between_unpriced_models():
    free_big = ModelProfile("aaa/big", "local", strengths=("cheap",), rank=9)
    free_small = ModelProfile("zzz/small", "local", strengths=("cheap",), rank=1)
    assert ModelRegistry([free_big, free_small]).cheapest_with("cheap").id == "zzz/small"


def test_cost_still_beats_rank():
    priced = ModelProfile("p/priced", "vendor", strengths=("cheap",), cost_per_1k_in=0.0001, rank=99)
    free = ModelProfile("f/free", "local", strengths=("cheap",), rank=1)
    assert ModelRegistry([priced, free]).cheapest_with("cheap").id == "f/free"

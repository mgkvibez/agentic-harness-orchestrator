import json
import pytest
from agent_harness.errors import MissingDependency, ProviderError
from agent_harness.providers.base import Message, estimate_tokens
from agent_harness.providers.fake import FakeProvider, ScriptedReply


def test_message_role_is_validated():
    with pytest.raises(ValueError):
        Message("wizard", "hi")


def test_estimate_tokens_is_monotonic():
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 40)


def test_fake_provider_matches_by_substring():
    provider = FakeProvider([{"contains": "route", "text": '{"route": "direct"}'}])
    reply = provider.complete([Message("user", "decide the route")], model="m")
    assert reply.text == '{"route": "direct"}'


def test_fake_provider_falls_back():
    provider = FakeProvider([{"contains": "nothing matches this", "text": "x"}], fallback="fb")
    assert provider.complete([Message("user", "hello")], model="m").text == "fb"


def test_fake_provider_records_calls_deterministically():
    provider = FakeProvider([{"contains": "", "text": "same"}])
    a = provider.complete([Message("user", "hi")], model="m")
    b = provider.complete([Message("user", "hi")], model="m")
    assert a.input_tokens == b.input_tokens and len(provider.calls) == 2


def test_fake_provider_needs_a_user_message():
    with pytest.raises(ProviderError):
        FakeProvider().complete([Message("system", "only a system prompt")], model="m")


def test_fake_provider_labels_itself_in_raw():
    assert FakeProvider().complete([Message("user", "hi")], model="m").raw["provider"] == "fake"


def test_script_from_json_file(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"script": [{"contains": "x", "text": "y"}], "fallback": "z"}))
    assert FakeProvider.from_json_file(str(path)).fallback == "z"


def test_litellm_provider_refuses_clearly_when_absent():
    try:
        import litellm  # noqa: F401
        pytest.skip("litellm is installed here, so the refusal path cannot be hit")
    except ImportError:
        pass
    from agent_harness.providers.litellm_provider import LiteLLMProvider
    with pytest.raises(MissingDependency) as exc:
        LiteLLMProvider()
    assert "litellm" in str(exc.value)

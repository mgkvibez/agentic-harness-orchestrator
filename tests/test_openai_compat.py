import json
import pytest
from agent_harness.errors import ProviderError
from agent_harness.providers.base import Message
from agent_harness.providers.openai_compat import OpenAICompatibleProvider

MSGS = [Message("system", "be brief"), Message("user", "say hi")]


def ok(text="hi", usage=(11, 3)):
    return 200, {"model": "m", "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}}


def provider(responses, **kw):
    seen = []
    def transport(url, headers, body, timeout):
        seen.append({"url": url, "headers": headers, "body": json.loads(body)})
        item = responses[min(len(seen) - 1, len(responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return item
    p = OpenAICompatibleProvider("https://example.test/v1", "secret-key",
                                sleep=lambda s: None, transport=transport, **kw)
    p.seen = seen
    return p


def test_sends_the_expected_request():
    p = provider([ok()])
    p.complete(MSGS, model="nvidia/x", temperature=0.2, max_tokens=64)
    sent = p.seen[0]
    assert sent["url"] == "https://example.test/v1/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer secret-key"
    assert sent["body"]["model"] == "nvidia/x" and sent["body"]["max_tokens"] == 64
    assert sent["body"]["messages"][0] == {"role": "system", "content": "be brief"}


def test_parses_content_and_reported_usage():
    completion = provider([ok("hello", (11, 3))]).complete(MSGS, model="m")
    assert completion.text == "hello"
    assert (completion.input_tokens, completion.output_tokens) == (11, 3)
    assert completion.raw["usage_reported"] is True


def test_estimates_usage_when_the_provider_reports_none():
    p = provider([(200, {"choices": [{"message": {"content": "x" * 40}}]})])
    completion = p.complete(MSGS, model="m")
    assert completion.output_tokens == 10 and completion.raw["usage_reported"] is False


def test_rate_limit_is_retried_then_succeeds():
    p = provider([(429, {"error": {"message": "slow down"}}), ok("fine")])
    assert p.complete(MSGS, model="m").text == "fine"
    assert p.retries_used == 1 and p.calls == 2


def test_retries_are_bounded_and_the_error_is_explicit():
    p = provider([(503, {"error": {"message": "unavailable"}})] * 4, retries=1)
    with pytest.raises(ProviderError) as exc:
        p.complete(MSGS, model="m")
    assert "503" in str(exc.value) and p.calls == 2


def test_a_400_is_not_retried():
    p = provider([(400, {"error": {"message": "bad request"}})] * 3)
    with pytest.raises(ProviderError):
        p.complete(MSGS, model="m")
    assert p.calls == 1, "client errors must fail fast instead of burning quota"


def test_a_malformed_body_raises_provider_error():
    p = provider([(200, {"choices": []})])
    with pytest.raises(ProviderError) as exc:
        p.complete(MSGS, model="m")
    assert "shape" in str(exc.value)


def test_missing_credentials_are_refused_at_construction():
    with pytest.raises(ProviderError):
        OpenAICompatibleProvider("https://example.test", "")
    with pytest.raises(ProviderError):
        OpenAICompatibleProvider("", "k")


def test_trailing_slash_is_normalised():
    assert OpenAICompatibleProvider("https://example.test/v1/", "k").base_url == "https://example.test/v1"

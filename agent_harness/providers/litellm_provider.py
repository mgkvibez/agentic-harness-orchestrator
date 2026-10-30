"""The real path: LiteLLM behind the same interface, imported lazily.

Blueprint decision 1: the core never talks to a provider API directly, and it
never imports this module unless a real run is asked for.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from ..errors import MissingDependency, ProviderBadResponse, ProviderError
from .base import coerce_usage
from .base import Completion, Message, estimate_tokens


class LiteLLMProvider:
    name = "litellm"

    def __init__(self, api_key_map: Optional[Dict[str, str]] = None):
        try:
            import litellm  # noqa: F401
        except ImportError as exc:
            raise MissingDependency(
                "the litellm provider needs the optional extra: pip install '{}[litellm]'".format("agent-harness")
            ) from exc
        self._api_keys = api_key_map or {}
        self._litellm = litellm

    @property
    def attempt_ceiling(self) -> int:
        # litellm performs its own internal retries; quote for two attempts.
        return 2

    def complete(self, messages: Sequence[Message], model: str, temperature: float = 0.0,
                 max_tokens: int = 1024) -> Completion:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [m.as_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if model.split("/")[0] in self._api_keys:
            payload["api_key"] = self._api_keys[model.split("/")[0]]
        try:
            response = self._litellm.completion(**payload)
        except Exception as exc:                       # provider SDK errors are open-ended
            raise ProviderError("litellm call for {!r} failed: {}".format(model, exc)) from exc
        try:
            text = response["choices"][0]["message"]["content"] or ""
            usage = response.get("usage") or {}
            finish = response["choices"][0].get("finish_reason", "stop")
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderBadResponse(
                "litellm returned an unexpected shape: {}".format(exc)) from exc
        input_tokens, output_tokens, reported = coerce_usage(
            usage if isinstance(usage, dict) else None, text, messages)
        return Completion(
            text=text, model=model, input_tokens=input_tokens, output_tokens=output_tokens,
            finish_reason=finish,
            raw={"provider": self.name, "usage_reported": reported, "attempts": 1,
                 "retries": 0})

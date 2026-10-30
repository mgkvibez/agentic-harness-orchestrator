"""Any OpenAI-compatible endpoint, with no SDK and no dependency.

Covers NVIDIA NIM, Groq, Together, OpenRouter, vLLM, LM Studio and Ollama's
compatible route with one adapter and urllib from the standard library. The
transport is injectable so the retry and parsing logic is testable without a
network, which is otherwise the least-tested code in any client.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Optional, Sequence

from ..errors import ProviderError
from .base import Completion, Message, estimate_tokens

Transport = Callable[[str, Dict[str, str], bytes, float], "tuple[int, Dict[str, Any]]"]

RETRY_STATUSES = (408, 409, 425, 429, 500, 502, 503, 504)


def urllib_transport(url: str, headers: Dict[str, str], body: bytes, timeout: float):
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(payload or "{}")
        except ValueError:
            return exc.code, {"error": {"message": payload[:500]}}
    except urllib.error.URLError as exc:
        raise ProviderError("cannot reach {}: {}".format(url, exc.reason)) from exc


class OpenAICompatibleProvider:
    name = "openai-compat"

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 90.0, retries: int = 2,
                 sleep=time.sleep, transport: Optional[Transport] = None):
        if not base_url:
            raise ProviderError("base_url is required for the openai-compatible provider")
        if not api_key:
            raise ProviderError("an api key is required for the openai-compatible provider")
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self._sleep = sleep
        self._transport = transport or urllib_transport
        self.calls = 0
        self.retries_used = 0

    def complete(self, messages: Sequence[Message], model: str, temperature: float = 0.0,
                 max_tokens: int = 1024) -> Completion:
        body = json.dumps({"model": model,
                           "messages": [m.as_dict() for m in messages],
                           "temperature": temperature,
                           "max_tokens": max_tokens}).encode()
        headers = {"Authorization": "Bearer {}".format(self._api_key),
                   "Content-Type": "application/json",
                   "Accept": "application/json"}
        url = self.base_url + "/chat/completions"
        attempt = 0
        while True:
            self.calls += 1
            status, payload = self._transport(url, headers, body, self.timeout)
            if status == 200:
                return self._parse(payload, model, messages)
            message = self._error_message(payload)
            if status in RETRY_STATUSES and attempt < self.retries:
                attempt += 1
                self.retries_used += 1
                self._sleep(0.5 * (2 ** (attempt - 1)))
                continue
            raise ProviderError("{} returned {} for {!r}: {}".format(
                self.base_url, status, model, message))

    def _error_message(self, payload: Dict[str, Any]) -> str:
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            return str(error.get("message") or error)[:300]
        return str(payload)[:300]

    def _parse(self, payload: Dict[str, Any], model: str, messages: Sequence[Message]) -> Completion:
        try:
            choice = payload["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("unexpected response shape for {!r}: {}".format(model, exc)) from exc
        usage = payload.get("usage") or {}
        return Completion(
            text=text, model=payload.get("model") or model,
            input_tokens=int(usage.get("prompt_tokens") or estimate_tokens("".join(m.content for m in messages))),
            output_tokens=int(usage.get("completion_tokens") or estimate_tokens(text)),
            finish_reason=choice.get("finish_reason", "stop"),
            raw={"provider": self.name, "usage_reported": bool(usage)})

"""Any OpenAI-compatible endpoint, with no SDK and no dependency.

Covers NVIDIA NIM, Groq, Together, OpenRouter, vLLM, LM Studio and Ollama's
compatible route with one adapter and urllib from the standard library. The
transport is injectable so the retry and parsing logic is testable without a
network, which is otherwise the least-tested code in any client.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Optional, Sequence

from ..errors import (ProviderAuthError, ProviderBadResponse, ProviderError,
                      ProviderInvalidRequest, ProviderModelNotFound, ProviderRateLimit,
                      ProviderTimeout, ProviderUnavailable)
from .base import Completion, Message, coerce_usage, estimate_tokens

Transport = Callable[[str, Dict[str, str], bytes, float], "tuple[int, Dict[str, Any]]"]

RETRY_STATUSES = (408, 409, 425, 429, 500, 502, 503, 504)


def urllib_transport(url: str, headers: Dict[str, str], body: bytes, timeout: float):
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
            try:
                payload = json.loads(body or "{}")
            except ValueError as exc:
                # A 200 that is not JSON is a provider fault, not a crash. Keep a
                # short snippet for diagnosis; never echo headers or credentials.
                raise ProviderBadResponse(
                    "response was not JSON (HTTP {}): {} | first bytes: {!r}".format(
                        response.status, exc, body[:120])) from exc
            if not isinstance(payload, dict):
                raise ProviderBadResponse(
                    "response was a JSON {} rather than an object".format(type(payload).__name__))
            return response.status, payload
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(payload or "{}")
        except ValueError:
            return exc.code, {"error": {"message": payload[:500]}}
    except socket.timeout as exc:
        raise ProviderTimeout("{} timed out after {}s".format(url, timeout)) from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, socket.timeout):
            raise ProviderTimeout("{} timed out after {}s".format(url, timeout)) from exc
        raise ProviderUnavailable("cannot reach {}: {}".format(url, exc.reason)) from exc


def _error_for_status(status: int, message: str, model: str, base_url: str,
                      attempts: int) -> ProviderError:
    """One status code, one cause, one correct response."""
    detail = "{} returned {} for {!r} after {} attempt(s): {}".format(
        base_url, status, model, attempts, message)
    if status in (401, 403):
        return ProviderAuthError(detail)
    if status == 404:
        return ProviderModelNotFound(detail)
    if status == 429:
        return ProviderRateLimit(detail)
    if status in (400, 408, 409, 422, 425):
        return ProviderInvalidRequest(detail)
    if 500 <= status < 600:
        return ProviderUnavailable(detail)
    return ProviderError(detail)


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
        # Counters are process-wide totals for reporting. Attribution per call is
        # carried on the Completion, so two orchestrators sharing one provider
        # cannot corrupt each other's attempt counts.
        self._lock = threading.Lock()
        self.calls = 0
        self.retries_used = 0

    @property
    def attempt_ceiling(self) -> int:
        """Worst-case attempts for one logical call, so the budget can reserve them."""
        return self.retries + 1

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
            with self._lock:
                self.calls += 1
            try:
                status, payload = self._transport(url, headers, body, self.timeout)
            except (ProviderTimeout, ProviderUnavailable):
                if attempt < self.retries:
                    attempt += 1
                    with self._lock:
                        self.retries_used += 1
                    self._sleep(0.5 * (2 ** (attempt - 1)))
                    continue
                raise
            if status == 200:
                return self._parse(payload, model, messages, attempt + 1)
            message = self._error_message(payload)
            if status in RETRY_STATUSES and attempt < self.retries:
                attempt += 1
                with self._lock:
                    self.retries_used += 1
                self._sleep(0.5 * (2 ** (attempt - 1)))
                continue
            raise _error_for_status(status, message, model, self.base_url, attempt + 1)

    def _error_message(self, payload: Dict[str, Any]) -> str:
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            return str(error.get("message") or error)[:300]
        return str(payload)[:300]

    def _parse(self, payload: Dict[str, Any], model: str, messages: Sequence[Message],
               attempts: int = 1) -> Completion:
        try:
            choice = payload["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderBadResponse(
                "unexpected response shape for {!r}: {}".format(model, exc)) from exc
        input_tokens, output_tokens, reported = coerce_usage(
            payload.get("usage"), text, messages)
        return Completion(
            text=text, model=payload.get("model") or model,
            input_tokens=input_tokens, output_tokens=output_tokens,
            finish_reason=choice.get("finish_reason", "stop"),
            raw={"provider": self.name, "usage_reported": reported,
                 "attempts": attempts, "retries": attempts - 1})

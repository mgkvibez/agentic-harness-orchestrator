"""The contract every provider speaks. The core never sees a vendor SDK."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from ..errors import ProviderBadResponse


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant"):
            raise ValueError("message role must be system, user or assistant, got {!r}".format(self.role))

    def as_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = "stop"
    raw: Optional[Dict[str, Any]] = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def estimate_tokens(text: str) -> int:
    """Rough, deterministic, and used only when a provider reports nothing.

    Four characters per token is the usual rule of thumb. It is an estimate and
    the trace says so rather than presenting it as measured usage.
    """
    return max(1, len(text) // 4)


class Provider(Protocol):
    name: str

    def complete(self, messages: Sequence[Message], model: str,
                 temperature: float = 0.0, max_tokens: int = 1024) -> Completion:
        """One turn. Raises ProviderError on transport or shape problems."""


def coerce_usage(usage: Any, text: str, messages: Sequence["Message"]) -> "tuple[int, int, bool]":
    """Validate provider-reported token counts.

    Absent usage is estimated and flagged, because a provider that does not report
    usage is common and not an error. Present-but-malformed usage is refused: a
    negative count, a bool, a string or a NaN cannot be costed, and silently
    coercing it would produce a confident wrong number in the trace.
    """
    if usage is None:
        usage = {}
    if not isinstance(usage, dict):
        raise ProviderBadResponse(
            "usage was {!r}, not an object; cannot cost this call".format(type(usage).__name__))

    def field(name: str, fallback: int) -> "tuple[int, bool]":
        if name not in usage or usage[name] is None:
            return fallback, False
        value = usage[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProviderBadResponse(
                "usage.{0} was {1!r}; a token count must be a number".format(
                    name, type(value).__name__))
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise ProviderBadResponse("usage.{} was not a finite number".format(name))
        if value < 0:
            raise ProviderBadResponse("usage.{} was negative ({})".format(name, value))
        return int(value), True

    prompt_estimate = estimate_tokens("".join(m.content for m in messages))
    output_estimate = estimate_tokens(text)
    input_tokens, reported_in = field("prompt_tokens", prompt_estimate)
    output_tokens, reported_out = field("completion_tokens", output_estimate)
    return input_tokens, output_tokens, (reported_in and reported_out)

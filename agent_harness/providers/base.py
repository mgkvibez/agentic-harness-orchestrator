"""The contract every provider speaks. The core never sees a vendor SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence


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

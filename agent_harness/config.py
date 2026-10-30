"""Settings from the environment. Defaults are safe: no network, no spend."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    max_steps: int = 8
    max_cost_usd: float = 0.25
    max_seconds: float = 120.0
    blackboard_char_limit: int = 40000
    trace_path: str = ""
    provider: str = "fake"

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        base = cls(
            max_steps=_int("AHARNESS_MAX_STEPS", 8),
            max_cost_usd=_float("AHARNESS_MAX_COST_USD", 0.25),
            max_seconds=_float("AHARNESS_MAX_SECONDS", 120.0),
            blackboard_char_limit=_int("AHARNESS_BLACKBOARD_CHARS", 40000),
            trace_path=os.environ.get("AHARNESS_TRACE", ""),
            provider=os.environ.get("AHARNESS_PROVIDER", "fake"),
        )
        return cls(**{**base.__dict__, **overrides}) if overrides else base

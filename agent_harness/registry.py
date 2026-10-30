"""The model registry: what each model is good at, and what it costs.

Static profiles on purpose (blueprint decision 5). They are a starting point
that a human curates, not a benchmark this tool claims to have run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .errors import ConfigError

STRENGTHS = ("reasoning", "coding", "writing", "summarising", "extraction", "planning", "cheap")
SPEEDS = ("fast", "balanced", "slow")


@dataclass(frozen=True)
class ModelProfile:
    id: str
    provider: str
    label: str = ""
    strengths: Sequence[str] = field(default_factory=tuple)
    speed: str = "balanced"
    cost_per_1k_in: float = 0.0
    cost_per_1k_out: float = 0.0
    context: int = 0
    rank: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.id or "/" in self.id and not self.id.strip():
            raise ConfigError("model id must be a non-empty string")
        unknown = [s for s in self.strengths if s not in STRENGTHS]
        if unknown:
            raise ConfigError("model {!r}: unknown strength(s) {}".format(self.id, unknown))
        if self.speed not in SPEEDS:
            raise ConfigError("model {!r}: speed must be one of {}".format(self.id, list(SPEEDS)))
        if self.cost_per_1k_in < 0 or self.cost_per_1k_out < 0:
            raise ConfigError("model {!r}: cost cannot be negative".format(self.id))
    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return round(input_tokens / 1000 * self.cost_per_1k_in
                     + output_tokens / 1000 * self.cost_per_1k_out, 6)

    @property
    def is_free(self) -> bool:
        return self.cost_per_1k_in == 0 and self.cost_per_1k_out == 0


def load_profiles(path: str) -> List[ModelProfile]:
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ConfigError("{}: cannot read model profiles ({})".format(path, exc)) from None
    entries = raw["models"] if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise ConfigError("{}: expected a list of models or an object with a 'models' list".format(path))
    profiles = []
    for entry in entries:
        if not isinstance(entry, dict) or "id" not in entry:
            raise ConfigError("{}: every model needs an 'id'".format(path))
        profiles.append(ModelProfile(
            id=entry["id"], provider=entry.get("provider", "unknown"),
            label=entry.get("label", ""), strengths=tuple(entry.get("strengths", ())),
            speed=entry.get("speed", "balanced"),
            cost_per_1k_in=float(entry.get("cost_per_1k_in", 0.0)),
            cost_per_1k_out=float(entry.get("cost_per_1k_out", 0.0)),
            rank=int(entry.get("rank", 0)),
            context=int(entry.get("context", 0)), notes=entry.get("notes", "")))
    if not profiles:
        raise ConfigError("{}: registry is empty".format(path))
    ids = [p.id for p in profiles]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ConfigError("{}: duplicate model id(s) {}".format(path, sorted(duplicates)))
    return profiles


class ModelRegistry:
    def __init__(self, profiles: Iterable[ModelProfile]):
        self._by_id: Dict[str, ModelProfile] = {}
        for profile in profiles:
            if profile.id in self._by_id:
                raise ConfigError("duplicate model id {!r}".format(profile.id))
            self._by_id[profile.id] = profile
        if not self._by_id:
            raise ConfigError("registry is empty")

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self):
        return iter(self._by_id.values())

    def get(self, model_id: str) -> ModelProfile:
        try:
            return self._by_id[model_id]
        except KeyError:
            raise ConfigError("unknown model {!r}; known models: {}".format(
                model_id, sorted(self._by_id))) from None

    def ids(self) -> List[str]:
        return sorted(self._by_id)

    def cheapest_with(self, strength: str, exclude: Optional[Sequence[str]] = None) -> ModelProfile:
        """The smallest capable model wins: cost first, then speed, then id."""
        exclude = set(exclude or ())
        candidates = [p for p in self._by_id.values()
                      if strength in p.strengths and p.id not in exclude]
        if not candidates:
            raise ConfigError("no model advertises the {!r} strength".format(strength))
        order = {"fast": 0, "balanced": 1, "slow": 2}
        # Cost first, then an explicit human rank, then speed, then id. The rank
        # matters when several models are unpriced: without it the tie-break is
        # alphabetical order, which is not a routing policy.
        return sorted(candidates, key=lambda p: (p.cost_per_1k_in + p.cost_per_1k_out,
                                                 p.rank, order.get(p.speed, 9), p.id))[0]

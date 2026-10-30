"""Agent roster: a role is a model plus a system prompt plus declared outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .errors import ConfigError
from .registry import ModelRegistry


@dataclass(frozen=True)
class Agent:
    name: str
    role: str
    model: str
    system_prompt: str
    writes: Sequence[str] = field(default_factory=tuple)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ConfigError("agent name must be a non-empty string")
        if not self.model:
            raise ConfigError("agent {!r}: a model is required".format(self.name))
        if not self.system_prompt.strip():
            raise ConfigError("agent {!r}: a system prompt is required, or the role is not a role".format(self.name))
        if not self.writes:
            raise ConfigError("agent {!r}: declare at least one blackboard key it writes".format(self.name))


def load_roster(path: str, registry: Optional[ModelRegistry] = None) -> "Roster":
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ConfigError("{}: cannot read the agent roster ({})".format(path, exc)) from None
    entries = raw["agents"] if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise ConfigError("{}: expected a list of agents or an object with an 'agents' list".format(path))
    agents = []
    for entry in entries:
        if not isinstance(entry, dict) or "name" not in entry:
            raise ConfigError("{}: every agent needs a 'name'".format(path))
        agents.append(Agent(
            name=entry["name"], role=entry.get("role", entry["name"]),
            model=entry["model"], system_prompt=entry["system_prompt"],
            writes=tuple(entry.get("writes", ())), notes=entry.get("notes", "")))
    return Roster(agents, registry=registry)


class Roster:
    def __init__(self, agents: Sequence[Agent], registry: Optional[ModelRegistry] = None):
        self._agents: Dict[str, Agent] = {}
        for agent in agents:
            if agent.name in self._agents:
                raise ConfigError("duplicate agent name {!r}".format(agent.name))
            if registry is not None:
                registry.get(agent.model)          # raises ConfigError naming the bad reference
            self._agents[agent.name] = agent
        if not self._agents:
            raise ConfigError("roster is empty")

    def __len__(self) -> int:
        return len(self._agents)

    def __iter__(self):
        return iter(self._agents.values())

    def get(self, name: str) -> Agent:
        try:
            return self._agents[name]
        except KeyError:
            raise ConfigError("unknown agent {!r}; roster has {}".format(name, sorted(self._agents))) from None

    def names(self) -> List[str]:
        return sorted(self._agents)

    def summary(self) -> str:
        return "\n".join("- {} ({}): model {} writes {}".format(
            a.name, a.role, a.model, ", ".join(a.writes)) for a in self)

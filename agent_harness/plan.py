"""JSON contracts with the models. Nothing is trusted until it validates.

Two contracts: the triage decision and the plan. Both are parsed strictly and a
model that answers outside them gets one repair attempt, then a logged fallback.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .errors import ContractViolation
from .roster import Roster

ROUTES = ("direct", "decompose")


def extract_json(text: str) -> Any:
    """Pull the first JSON object or array out of model prose."""
    if text is None:
        raise ContractViolation("model returned nothing")
    candidate = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = candidate.find(opener), candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ContractViolation("no JSON found in model output: {!r}".format(text[:160]))


@dataclass(frozen=True)
class TriageDecision:
    route: str
    reason: str = ""
    agent: str = ""
    source: str = "model"          # "model" or "fallback", and the trace shows which

    def __post_init__(self) -> None:
        if self.route not in ROUTES:
            raise ContractViolation("triage route must be one of {}, got {!r}".format(list(ROUTES), self.route))


@dataclass(frozen=True)
class Step:
    id: str
    agent: str
    instruction: str
    reads: Sequence[str] = field(default_factory=tuple)
    writes: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.id or not self.agent or not self.instruction.strip():
            raise ContractViolation("a step needs an id, an agent and an instruction")
        if not self.writes:
            raise ContractViolation("step {!r} declares no output; a step that writes nothing cannot be audited".format(self.id))


@dataclass(frozen=True)
class Plan:
    steps: Sequence[Step]
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.steps:
            raise ContractViolation("a plan needs at least one step")
        ids = [s.id for s in self.steps]
        duplicated = {i for i in ids if ids.count(i) > 1}
        if duplicated:
            raise ContractViolation("duplicate step id(s): {}".format(sorted(duplicated)))


def parse_triage(text: str, roster: Roster) -> TriageDecision:
    payload = extract_json(text)
    if not isinstance(payload, dict):
        raise ContractViolation("triage must be a JSON object")
    route = str(payload.get("route", "")).strip().lower()
    if route not in ROUTES:
        raise ContractViolation("triage route must be one of {}, got {!r}".format(list(ROUTES), route))
    agent = str(payload.get("agent", "")).strip()
    if agent and agent not in roster.names():
        raise ContractViolation("triage named unknown agent {!r}".format(agent))
    if route == "direct" and not agent:
        raise ContractViolation("a direct route must name the agent to run")
    return TriageDecision(route=route, reason=str(payload.get("reason", ""))[:400], agent=agent)


def parse_plan(text: str, roster: Roster, max_steps: int) -> Plan:
    payload = extract_json(text)
    if isinstance(payload, dict):
        steps_raw = payload.get("steps")
        rationale = str(payload.get("rationale", ""))[:400]
    else:
        steps_raw = payload
        rationale = ""
    if not isinstance(steps_raw, list):
        raise ContractViolation("plan must be an object with a 'steps' list")
    if len(steps_raw) > max_steps:
        raise ContractViolation("plan has {} steps, over the {}-step ceiling".format(len(steps_raw), max_steps))
    steps: List[Step] = []
    for index, raw in enumerate(steps_raw):
        if not isinstance(raw, dict):
            raise ContractViolation("step {} is not an object".format(index))
        agent = str(raw.get("agent", "")).strip()
        if agent not in roster.names():
            raise ContractViolation("step {} names unknown agent {!r}; roster has {}".format(
                index, agent, roster.names()))
        writes = raw.get("writes") or []
        if isinstance(writes, str):
            writes = [writes]
        reads = raw.get("reads") or []
        if isinstance(reads, str):
            reads = [reads]
        steps.append(Step(id=str(raw.get("id") or "step-{}".format(index + 1)),
                          agent=agent,
                          instruction=str(raw.get("instruction") or "").strip(),
                          reads=tuple(str(r) for r in reads),
                          writes=tuple(str(w) for w in writes)))
    return Plan(steps=steps, rationale=rationale)

"""JSON contracts with the models, and the dependency rules over a plan.

Two contracts (triage, plan) plus a validator that runs before anything is spent
on execution: write authority is checked against the roster, references must
resolve, cycles are refused, and the execution order is derived rather than
trusted. Every rejection is a ContractViolation so the repair path can engage.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .errors import ContractViolation, PermissionDenied
from .permissions import WritePolicy, is_reserved
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
    source: str = "model"

    def __post_init__(self) -> None:
        if self.route not in ROUTES:
            raise ContractViolation("triage route must be one of {}, got {!r}".format(
                list(ROUTES), self.route))


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
            raise ContractViolation("step {!r} declares no output; a step that writes nothing "
                                    "cannot be audited".format(self.id))


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


@dataclass(frozen=True)
class ValidatedPlan:
    """A plan that has been proved executable, plus what had to change."""
    steps: Sequence[Step]
    reordered: bool = False
    declared_order: Sequence[str] = field(default_factory=tuple)
    notes: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.steps:
            raise ContractViolation("a validated plan cannot be empty")


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
        raise ContractViolation("plan has {} steps, over the {}-step ceiling".format(
            len(steps_raw), max_steps))
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


def validate_plan(plan: Plan, roster: Roster, preseeded: Iterable[str],
                  policy: Optional[WritePolicy] = None, max_steps: int = 8) -> ValidatedPlan:
    """Prove the plan is executable before a single paid call is made."""
    if len(plan.steps) > max_steps:
        raise ContractViolation("plan has {} steps, over the {}-step ceiling".format(
            len(plan.steps), max_steps))
    notes: List[str] = []
    available: Set[str] = set(preseeded)

    # 1. write authority comes from the roster, never from the plan.
    for step in plan.steps:
        for key in step.writes:
            if is_reserved(key):
                raise ContractViolation(
                    "step {!r} tries to write reserved system key {!r}".format(step.id, key))
            if policy is not None:
                try:
                    policy.check(step.agent, key)
                except PermissionDenied as exc:
                    raise ContractViolation(
                        "step {!r}: {}. A plan cannot grant authority the roster does not."
                        .format(step.id, exc)) from None

    # 2. every read must resolve to something already supplied or produced.
    produced_by: Dict[str, str] = {}
    for step in plan.steps:
        for key in step.writes:
            if key in produced_by and produced_by[key] != step.id:
                notes.append("key {!r} is written by both {!r} and {!r}".format(
                    key, produced_by[key], step.id))
            produced_by.setdefault(key, step.id)
    for step in plan.steps:
        for key in step.reads:
            if key in available or key in produced_by:
                continue
            raise ContractViolation(
                "step {!r} reads {!r}, which no step produces and which was not supplied"
                .format(step.id, key))

    # 3. dependencies, and refusal of cycles.
    deps: Dict[str, Set[str]] = {step.id: set() for step in plan.steps}
    for step in plan.steps:
        for key in step.reads:
            writer = produced_by.get(key)
            if writer and writer != step.id:
                deps[step.id].add(writer)
    _reject_cycles(deps)

    # 4. deterministic order: stable topological sort, declared order preserved
    #    among steps that do not depend on each other.
    order = _topological_order(plan.steps, deps)
    declared = [s.id for s in plan.steps]
    reordered = [s.id for s in order] != declared
    if reordered:
        notes.append("declared order {} does not satisfy dependencies; executing {}".format(
            declared, [s.id for s in order]))
    return ValidatedPlan(steps=order, reordered=reordered, declared_order=declared,
                         notes=notes)


def _reject_cycles(deps: Dict[str, Set[str]]) -> None:
    WHITE, GREY, BLACK = 0, 1, 2
    colour = {node: WHITE for node in deps}
    stack: List[str] = []

    def visit(node: str) -> None:
        colour[node] = GREY
        stack.append(node)
        for dep in sorted(deps.get(node, ())):
            if colour.get(dep, WHITE) == GREY:
                cycle = stack[stack.index(dep):] + [dep]
                raise ContractViolation("circular dependency: {}".format(" -> ".join(cycle)))
            if colour.get(dep, WHITE) == WHITE:
                visit(dep)
        stack.pop()
        colour[node] = BLACK

    for node in sorted(deps):
        if colour[node] == WHITE:
            visit(node)


def _topological_order(steps: Sequence[Step], deps: Dict[str, Set[str]]) -> List[Step]:
    index = {step.id: i for i, step in enumerate(steps)}
    remaining = {step.id: set(deps.get(step.id, ())) for step in steps}
    ordered: List[Step] = []
    while remaining:
        ready = [sid for sid, needs in remaining.items() if not needs]
        if not ready:                       # unreachable: cycles are rejected first
            raise ContractViolation("dependency graph cannot be ordered")
        ready.sort(key=lambda sid: index[sid])
        chosen = ready[0]
        ordered.append(steps[index[chosen]])
        remaining.pop(chosen)
        for needs in remaining.values():
            needs.discard(chosen)
    return ordered

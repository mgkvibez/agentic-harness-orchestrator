"""The harness: triage, plan, prove, reserve, execute, deliver, explain.

Order of operations is the substance of Phase 0. Nothing is called before it is
quoted and reserved, nothing executes before its dependencies are proved present,
and no write happens outside the roster's authority. Every refusal is an event in
the trace, so a failed run is still reconstructable.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .blackboard import ACTOR_SYSTEM, ACTOR_USER, Blackboard
from .budget import Budget
from .config import Settings
from .errors import (BudgetExceeded, ConfigError, ContractViolation, DependencyError,
                     HarnessError, OverwriteRejected, PermissionDenied, ProviderError)
from .permissions import WritePolicy
from .plan import (Plan, Step, TriageDecision, ValidatedPlan, extract_json, parse_plan,
                   parse_triage, validate_plan)
from .providers.base import Message, estimate_tokens
from .registry import ModelRegistry
from .roster import Agent, Roster
from .trace import Trace

TRIAGE_SYSTEM = (
    "You are the triage phase of an orchestration harness. Decide whether the task needs one agent "
    "or several. Answer with JSON only, no prose: "
    '{"route": "direct" | "decompose", "agent": "<agent name, required when direct>", "reason": "<short>"}'
)

PLANNER_SYSTEM = (
    "You are the planning phase. Break the task into the smallest number of ordered steps that will "
    "finish it. Each step names one agent from the roster and the blackboard keys it reads and writes. "
    'Answer with JSON only: {"rationale": "<short>", "steps": [{"id": "s1", "agent": "<name>", '
    '"instruction": "<what to do>", "reads": ["<key>"], "writes": ["<key>"]}]}'
)

STEP_FORMAT = (
    "Reply with the content for the key(s) you were asked to write. If a structured value is expected, "
    "reply with JSON only. Do not describe what you would do; do the work."
)

REPAIR = ("Your previous reply was not usable: {error}. "
          "Return only the JSON described in the instructions, with no commentary or code fences.")


@dataclass
class RunResult:
    task: str
    deliverable: Any
    route: str
    triage_source: str
    trace: Trace
    board: Dict[str, Any]
    cost_usd: float
    steps_run: int
    run_id: str = ""
    errors: List[str] = field(default_factory=list)
    budget: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "delivered"

    def as_dict(self) -> Dict[str, Any]:
        return {"run_id": self.run_id, "task": self.task, "route": self.route,
                "triage_source": self.triage_source, "deliverable": self.deliverable,
                "cost_usd": self.cost_usd, "steps_run": self.steps_run,
                "board_keys": sorted(self.board), "budget": self.budget,
                "artifacts": self.artifacts, "status": self.status,
                "errors": self.errors}


class Orchestrator:
    def __init__(self, registry: ModelRegistry, roster: Roster, provider,
                 settings: Optional[Settings] = None, trace: Optional[Trace] = None,
                 clock=time.monotonic, run_id: str = ""):
        self.registry = registry
        self.roster = roster
        self.provider = provider
        self.settings = settings or Settings.from_env()
        # Authority is derived from the roster, once, and cannot be widened later.
        self.policy = WritePolicy(roster)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self._trace_supplied = trace is not None
        self.trace = trace if trace is not None else Trace(self.settings.trace_path, self.run_id)
        self.trace.run_id = self.run_id
        self.clock = clock
        self.budget = Budget(limit_usd=self.settings.max_cost_usd,
                             unknown_pricing_per_1k=self.settings.unknown_pricing_per_1k)
        self._started = 0.0
        self.board: Optional[Blackboard] = None

    # -- helpers -------------------------------------------------------------
    def _cheap_model(self) -> str:
        try:
            return self.registry.cheapest_with("cheap").id
        except ConfigError:
            try:
                return self.registry.cheapest_with("summarising").id
            except ConfigError:
                raise ConfigError(
                    "no model is tagged 'cheap' or 'summarising', so triage has nothing to route with. "
                    "Tag the smallest model in your registry: routing a trivial decision to the flagship "
                    "defeats the point of the harness.") from None

    def _elapsed(self) -> float:
        return self.clock() - self._started

    def _check_wall_clock(self, label: str) -> None:
        if self._elapsed() > self.settings.max_seconds:
            raise BudgetExceeded("wall clock exceeded ({:.1f}s > {:.1f}s) before {}".format(
                self._elapsed(), self.settings.max_seconds, label), trace=self.trace)

    def _attempt_ceiling(self) -> int:
        return max(1, int(getattr(self.provider, "attempt_ceiling", 1)))

    # -- one provider call: quote, reserve, call, settle, record --------------
    def _ask(self, agent_name: str, model: str, system: str, prompt: str, *,
             step_id: str = "", max_tokens: Optional[int] = None, phase: str = ""):
        profile = self.registry.get(model)
        budget_tokens = max_tokens or self.settings.max_output_tokens
        quote = self.budget.quote_for_prompt(profile, system + "\n" + prompt, budget_tokens,
                                             attempts=self._attempt_ceiling())
        reservation = self.budget.reserve(
            quote, label="{} call to {} for {!r}".format(phase or "provider", model, agent_name))
        call_id = uuid.uuid4().hex[:10]
        messages = [Message("system", system), Message("user", prompt)]
        started = self.clock()
        retries_before = int(getattr(self.provider, "retries_used", 0) or 0)
        try:
            completion = self.provider.complete(messages, model=model, max_tokens=budget_tokens)
        except HarnessError:
            # The call was reserved but produced nothing billable we can measure.
            self.budget.release(reservation)
            self.trace.record("provider_call_failed", step_id=step_id, agent=agent_name,
                              model=model, phase=phase, call_id=call_id,
                              quote_usd=quote.usd, conservative=quote.conservative)
            raise
        duration_ms = int((self.clock() - started) * 1000)
        if profile.pricing_known:
            actual = profile.cost(completion.input_tokens, completion.output_tokens)
        else:
            actual, _ = profile.estimate_cost(completion.input_tokens, completion.output_tokens,
                                              self.settings.unknown_pricing_per_1k)
        self.budget.settle(reservation, actual)
        attempts = 1 + (int(getattr(self.provider, "retries_used", 0) or 0) - retries_before)
        self.trace.record("step_end", step_id=step_id, agent=agent_name, model=model,
                          phase=phase, call_id=call_id,
                          input_tokens=completion.input_tokens,
                          output_tokens=completion.output_tokens,
                          cost_usd=actual, estimated_cost_usd=quote.usd,
                          conservative=quote.conservative,
                          budget_remaining_usd=self.budget.remaining_usd,
                          duration_ms=duration_ms, attempts=attempts,
                          finish_reason=completion.finish_reason,
                          estimated=bool((completion.raw or {}).get("provider") == "fake"))
        return completion

    def _ask_json(self, agent_name: str, model: str, system: str, prompt: str, parser,
                  step_id: str, phase: str):
        """Ask, validate, buy room if truncated, repair once, then let the caller fall back."""
        completion = self._ask(agent_name, model, system, prompt, step_id=step_id, phase=phase)
        if completion.finish_reason == "length":
            bigger = (self.settings.max_output_tokens * 3
                      if self.settings.max_output_tokens else 3072)
            self.trace.record("truncated", step_id=step_id, agent=agent_name,
                              first_budget=self.settings.max_output_tokens, retry_budget=bigger)
            completion = self._ask(agent_name, model, system, prompt, step_id=step_id,
                                   max_tokens=bigger, phase=phase)
        try:
            decision = parser(completion.text)
            self.trace.record("validation", step_id=step_id, agent=agent_name, phase=phase,
                              result="ok")
            return decision, completion
        except ContractViolation as first_error:
            self.trace.record("contract_violation", step_id=step_id, agent=agent_name,
                              phase=phase, error=str(first_error)[:300])
            repair_prompt = prompt + "\n\n" + REPAIR.format(error=str(first_error)[:200])
            completion = self._ask(agent_name, model, system, repair_prompt, step_id=step_id,
                                   phase=phase)
            decision = parser(completion.text)
            self.trace.record("validation", step_id=step_id, agent=agent_name, phase=phase,
                              result="ok_after_repair")
            return decision, completion

    # -- public entry points -------------------------------------------------
    def triage(self, task: str) -> TriageDecision:
        """Routing only: what would this task be sent to, and why. No steps run."""
        self._begin(task, {})
        assert self.board is not None
        decision = self._triage(task, self.board)
        self.trace.record("triage", route=decision.route, agent=decision.agent,
                          source=decision.source, reason=decision.reason)
        self.trace.record("run_end", status="triage_only", cost_usd=self.budget.spent_usd,
                          estimated_usd=self.budget.estimated_usd, steps=0,
                          budget=self.budget.snapshot(), elapsed_s=round(self._elapsed(), 3))
        return decision

    def run(self, task: str, inputs: Optional[Dict[str, Any]] = None) -> RunResult:
        board = self._begin(task, inputs or {})
        route, triage_source = "", ""
        steps_run = 0
        try:
            decision = self._triage(task, board)
            route, triage_source = decision.route, decision.source
            self.trace.record("triage", route=decision.route, agent=decision.agent,
                              source=decision.source, reason=decision.reason)

            if decision.route == "direct":
                raw_plan = Plan(steps=[Step(id="direct", agent=decision.agent,
                                            instruction=task.strip(),
                                            writes=(self._default_write(decision.agent),))],
                                rationale="triage chose the direct route")
            else:
                raw_plan = self._plan(task, board)
            preseeded = set(board.keys())
            validated = validate_plan(raw_plan, self.roster, preseeded,
                                      policy=self.policy, max_steps=self.settings.max_steps)
            self.trace.record("plan", steps=len(validated.steps), rationale=raw_plan.rationale,
                              step_ids=[s.id for s in validated.steps],
                              declared_order=list(validated.declared_order),
                              reordered=validated.reordered, notes=list(validated.notes))

            steps_run = len(validated.steps)
            self._execute(validated, board)
            deliverable, deliver_key = self._deliver(board, validated)
            self.trace.record("deliver", key=deliver_key,
                              value_type=type(deliverable).__name__,
                              sha256=self._artifact_sha(board, deliver_key),
                              status="delivered")
            self.trace.record("run_end", status="delivered", cost_usd=self.budget.spent_usd,
                              estimated_usd=self.budget.estimated_usd,
                              steps=steps_run, artifacts=len(self.trace.artifacts()),
                              budget=self.budget.snapshot(), elapsed_s=round(self._elapsed(), 3))
            return RunResult(task=task.strip(), deliverable=deliverable, route=route,
                             triage_source=triage_source, trace=self.trace,
                             board=board.snapshot(), cost_usd=self.budget.spent_usd,
                             steps_run=steps_run, run_id=self.run_id,
                             budget=self.budget.snapshot(), artifacts=self.trace.artifacts(),
                             status="delivered")
        except HarnessError as exc:
            status = ("budget_exceeded" if isinstance(exc, BudgetExceeded) else
                      "refused" if isinstance(exc, (PermissionDenied, OverwriteRejected,
                                                    DependencyError)) else "failed")
            self.trace.record("run_end", status=status, error=type(exc).__name__,
                              error_detail=str(exc)[:400], cost_usd=self.budget.spent_usd,
                              estimated_usd=self.budget.estimated_usd,
                              steps=steps_run, artifacts=len(self.trace.artifacts()),
                              budget=self.budget.snapshot(), elapsed_s=round(self._elapsed(), 3))
            if isinstance(exc, BudgetExceeded) and exc.trace is None:
                exc.trace = self.trace
            raise

    # -- internals -----------------------------------------------------------
    def _begin(self, task: str, inputs: Dict[str, Any]) -> Blackboard:
        if not task or not task.strip():
            raise ConfigError("a task is required")
        self._started = self.clock()
        self.budget = Budget(limit_usd=self.settings.max_cost_usd,
                             unknown_pricing_per_1k=self.settings.unknown_pricing_per_1k)
        # A caller-supplied trace is kept, not replaced: evidence recorded before
        # the run (credentials, caller context) must survive into the same file.
        if not self._trace_supplied:
            self.trace = Trace(self.settings.trace_path, self.run_id)
        trace = self.trace
        board = Blackboard(char_limit=self.settings.blackboard_char_limit, policy=self.policy)
        board.write("task", task.strip(), written_by="harness", actor=ACTOR_SYSTEM)
        for key, value in inputs.items():
            board.write(key, value, written_by="user", actor=ACTOR_USER)
        board.close_seeding()
        self.board = board
        trace.record("run_start", task=task.strip()[:300],
                     provider=getattr(self.provider, "name", "unknown"),
                     models=len(self.registry), agents=len(self.roster),
                     budget=self.budget.snapshot(),
                     policy={name: sorted(self.policy.allowed_keys(name))
                             for name in self.roster.names()})
        return board

    def _triage(self, task: str, board: Blackboard) -> TriageDecision:
        model = self._cheap_model()
        prompt = "Task: {}\n\nAgents available:\n{}\n\nDecide the route.".format(
            task.strip(), self.roster.summary())
        try:
            decision, _ = self._ask_json("triage", model, TRIAGE_SYSTEM, prompt,
                                         lambda text: parse_triage(text, self.roster),
                                         step_id="triage", phase="triage")
            return decision
        except ContractViolation as exc:
            fallback = self._fallback_agent()
            self.trace.record("fallback", phase="triage", reason=str(exc)[:300], agent=fallback)
            return TriageDecision(route="direct", agent=fallback,
                                  reason="triage contract failed", source="fallback")

    def _plan(self, task: str, board: Blackboard) -> Plan:
        model = self._cheap_model()
        prompt = ("Task: {}\n\nAgents available:\n{}\n\nProduce the plan. At most {} steps.".format(
            task.strip(), self.roster.summary(), self.settings.max_steps))
        try:
            plan, _ = self._ask_json("planner", model, PLANNER_SYSTEM, prompt,
                                     lambda text: parse_plan(text, self.roster,
                                                             self.settings.max_steps),
                                     step_id="plan", phase="plan")
            return plan
        except ContractViolation as exc:
            agent = self._fallback_agent()
            self.trace.record("fallback", phase="plan", reason=str(exc)[:300], agent=agent)
            return Plan(steps=[Step(id="fallback-1", agent=agent, instruction=task.strip(),
                                    writes=(self._default_write(agent),))],
                        rationale="planner contract failed, one step fallback")

    def _execute(self, plan: ValidatedPlan, board: Blackboard) -> None:
        for index, step in enumerate(plan.steps):
            self._check_wall_clock("step {}".format(step.id))
            if index >= self.settings.max_steps:
                raise BudgetExceeded("step ceiling of {} reached".format(self.settings.max_steps),
                                     trace=self.trace)
            # Dependencies are proved present, never assumed.
            missing = board.missing(step.reads)
            if missing:
                self.trace.record("dependency_unmet", step_id=step.id, agent=step.agent,
                                  missing=missing, available=board.keys())
                raise DependencyError(
                    "step {!r} requires {} but the board holds {}".format(
                        step.id, missing, board.keys()), step_id=step.id, missing=missing)
            agent = self.roster.get(step.agent)
            available = board.slice(step.reads)
            prompt_parts = ["Task: {}".format(board.read("task", "")), "",
                            "Your step: {}".format(step.instruction)]
            if available:
                prompt_parts += ["", "Shared state you were told to read:",
                                 json.dumps(available, indent=2, sort_keys=True)[:12000]]
            prompt_parts += ["", "Write: {}".format(", ".join(step.writes)), STEP_FORMAT]
            self.trace.record("step_start", step_id=step.id, agent=step.agent, model=agent.model,
                              reads=list(step.reads), writes=list(step.writes))
            try:
                completion = self._ask(step.agent, agent.model, agent.system_prompt,
                                       "\n".join(prompt_parts), step_id=step.id, phase="step")
                if completion.finish_reason == "length":
                    bigger = (self.settings.max_output_tokens * 3
                              if self.settings.max_output_tokens else 3072)
                    self.trace.record("truncated", step_id=step.id, agent=step.agent,
                                      first_budget=self.settings.max_output_tokens,
                                      retry_budget=bigger)
                    completion = self._ask(step.agent, agent.model, agent.system_prompt,
                                           "\n".join(prompt_parts), step_id=step.id,
                                           max_tokens=bigger, phase="step")
                if not (completion.text or "").strip():
                    self.trace.record("empty_write", step_id=step.id, agent=step.agent,
                                      finish_reason=completion.finish_reason)
                    raise ContractViolation(
                        "step {!r} produced no content (finish_reason={!r})".format(
                            step.id, completion.finish_reason))
            except ProviderError as exc:
                self.trace.record("step_error", step_id=step.id, error=str(exc)[:300])
                raise
            self._write_outputs(step, completion.text, board)

    def _write_outputs(self, step: Step, text: str, board: Blackboard) -> None:
        """Several declared outputs mean several artifacts, not one value copied."""
        value: Any = text
        try:
            value = extract_json(text)
        except ContractViolation:
            pass
        if len(step.writes) == 1:
            assignments = {step.writes[0]: value}
        elif isinstance(value, dict) and all(key in value for key in step.writes):
            assignments = {key: value[key] for key in step.writes}
        elif isinstance(value, dict):
            missing = [key for key in step.writes if key not in value]
            self.trace.record("validation", step_id=step.id, agent=step.agent, result="failed",
                              reason="declared outputs missing from reply", missing=missing)
            raise ContractViolation(
                "step {!r} declares outputs {} but the reply only contained {}".format(
                    step.id, list(step.writes), sorted(value)))
        else:
            raise ContractViolation(
                "step {!r} declares {} outputs but returned a single value; reply with a JSON "
                "object keyed by output name".format(step.id, len(step.writes)))
        for key, payload in assignments.items():
            try:
                entry = board.write(key, payload, written_by=step.agent, step_id=step.id)
            except (PermissionDenied, OverwriteRejected) as exc:
                self.trace.record("write_rejected", step_id=step.id, agent=step.agent,
                                  key=key, reason=type(exc).__name__, detail=str(exc)[:300])
                raise
            self.trace.record("artifact_written", step_id=step.id, agent=step.agent, key=key,
                              sha256=entry.sha256, chars=entry.chars, version=entry.version)

    def _deliver(self, board: Blackboard, plan: ValidatedPlan):
        for candidate in ("deliverable", "report", "summary", "output"):
            if board.has(candidate):
                return board.read(candidate), candidate
        last_written = board.entries()[-1].key if board.entries() else None
        if last_written is None:
            raise ConfigError("no step wrote anything, so there is nothing to deliver")
        return board.read(last_written), last_written

    def _artifact_sha(self, board: Blackboard, key: str) -> str:
        for entry in reversed(board.entries()):
            if entry.key == key:
                return entry.sha256
        return ""

    def _default_write(self, agent_name: str) -> str:
        agent = self.roster.get(agent_name)
        for candidate in ("deliverable", "report", "summary"):
            if candidate in agent.writes:
                return candidate
        return agent.writes[0]

    def _fallback_agent(self) -> str:
        for agent in self.roster:
            if "deliverable" in agent.writes:
                return agent.name
        return self.roster.names()[0]

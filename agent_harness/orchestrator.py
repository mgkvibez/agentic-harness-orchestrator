"""The harness: triage, plan, run the board, hand back the work and the receipts.

A while loop over a blackboard, as the blueprint asked. The interesting parts are
the bounds: every model answer is validated, every step is budgeted, and a run
that goes wrong still returns its trace instead of vanishing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .blackboard import Blackboard
from .config import Settings
from .errors import BudgetExceeded, ConfigError, ContractViolation, ProviderError
from .plan import Plan, Step, TriageDecision, extract_json, parse_plan, parse_triage
from .providers.base import Message, Provider
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
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"task": self.task, "route": self.route, "triage_source": self.triage_source,
                "deliverable": self.deliverable, "cost_usd": self.cost_usd,
                "steps_run": self.steps_run, "board_keys": sorted(self.board),
                "errors": self.errors}


class Orchestrator:
    def __init__(self, registry: ModelRegistry, roster: Roster, provider: Provider,
                 settings: Optional[Settings] = None, trace: Optional[Trace] = None,
                 clock=time.monotonic):
        self.registry = registry
        self.roster = roster
        self.provider = provider
        self.settings = settings or Settings.from_env()
        self.trace = trace if trace is not None else Trace(self.settings.trace_path)
        self.clock = clock
        self._started = 0.0
        self._spent = 0.0

    # helpers ----------------------------------------------------------------
    def _cheap_model(self) -> str:
        """Triage and planning go to the smallest capable model, never the flagship."""
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

    def _check_budget(self, next_step: str) -> None:
        if self._elapsed() > self.settings.max_seconds:
            raise BudgetExceeded("wall clock exceeded ({:.1f}s > {:.1f}s) before {}".format(
                self._elapsed(), self.settings.max_seconds, next_step), trace=self.trace)
        if self._spent > self.settings.max_cost_usd:
            raise BudgetExceeded("cost ceiling exceeded (${:.5f} > ${:.5f}) before {}".format(
                self._spent, self.settings.max_cost_usd, next_step), trace=self.trace)

    def _ask(self, agent_name: str, model: str, system: str, prompt: str,
             step_id: str = "", max_tokens: Optional[int] = None) -> "object":
        messages = [Message("system", system), Message("user", prompt)]
        started = self.clock()
        completion = self.provider.complete(
            messages, model=model,
            max_tokens=max_tokens or self.settings.max_output_tokens)
        duration_ms = int((self.clock() - started) * 1000)
        profile = self.registry.get(model)
        cost = profile.cost(completion.input_tokens, completion.output_tokens)
        self._spent = round(self._spent + cost, 6)
        self.trace.record("step_end", step_id=step_id, agent=agent_name, model=model,
                          input_tokens=completion.input_tokens,
                          output_tokens=completion.output_tokens,
                          cost_usd=cost, duration_ms=duration_ms,
                          finish_reason=completion.finish_reason,
                          estimated=bool((completion.raw or {}).get("provider") == "fake"))
        return completion, duration_ms

    def _ask_json(self, agent_name: str, model: str, system: str, prompt: str,
                  parser, step_id: str):
        """Ask, validate, and repair once before giving up. Fallbacks are logged."""
        completion, _ = self._ask(agent_name, model, system, prompt, step_id=step_id)
        if completion.finish_reason == "length":
            # Reasoning models spend output budget on hidden thinking, so a small
            # ceiling truncates the answer and looks exactly like a model that
            # ignored the contract. It is a budget problem, so buy more room once.
            bigger = (self.settings.max_output_tokens * 3
                      if self.settings.max_output_tokens else 3072)
            self.trace.record("truncated", step_id=step_id, agent=agent_name,
                              first_budget=self.settings.max_output_tokens, retry_budget=bigger)
            completion, _ = self._ask(agent_name, model, system, prompt,
                                      step_id=step_id, max_tokens=bigger)
        try:
            return parser(completion.text), completion
        except ContractViolation as first_error:
            self.trace.record("contract_violation", step_id=step_id, agent=agent_name,
                              error=str(first_error)[:300])
            repair_prompt = prompt + "\n\n" + REPAIR.format(error=str(first_error)[:200])
            completion, _ = self._ask(agent_name, model, system, repair_prompt, step_id=step_id)
            return parser(completion.text), completion

    def triage(self, task: str) -> TriageDecision:
        """Routing only: what would this task be sent to, and why. No steps run."""
        if not task or not task.strip():
            raise ConfigError("a task is required")
        self._started = self.clock()
        self._spent = 0.0
        board = Blackboard(char_limit=self.settings.blackboard_char_limit)
        board.write("task", task.strip(), written_by="user")
        decision = self._triage(task, board)
        self.trace.record("triage", route=decision.route, agent=decision.agent,
                          source=decision.source, reason=decision.reason)
        return decision

    # the run ----------------------------------------------------------------
    def run(self, task: str, inputs: Optional[Dict[str, Any]] = None) -> RunResult:
        if not task or not task.strip():
            raise ConfigError("a task is required")
        self._started = self.clock()
        self._spent = 0.0
        board = Blackboard(char_limit=self.settings.blackboard_char_limit)
        board.write("task", task.strip(), written_by="user")
        for key, value in (inputs or {}).items():
            board.write(key, value, written_by="user")
        self.trace.record("run_start", task=task.strip()[:300], provider=getattr(self.provider, "name", "unknown"),
                          models=len(self.registry), agents=len(self.roster))
        errors: List[str] = []

        decision = self._triage(task, board)
        self.trace.record("triage", route=decision.route, agent=decision.agent,
                          source=decision.source, reason=decision.reason)

        if decision.route == "direct":
            plan = Plan(steps=[Step(id="direct", agent=decision.agent, instruction=task.strip(),
                                    writes=(self._default_write(decision.agent),))],
                        rationale="triage chose the direct route")
        else:
            plan = self._plan(task, board)
        self.trace.record("plan", steps=len(plan.steps), rationale=plan.rationale,
                          step_ids=[s.id for s in plan.steps])

        self._execute(plan, board)
        deliverable, deliver_key = self._deliver(board, plan)
        self.trace.record("deliver", key=deliver_key, value_type=type(deliverable).__name__)
        self.trace.record("run_end", cost_usd=self._spent, steps=len(plan.steps),
                          elapsed_s=round(self._elapsed(), 3))
        return RunResult(task=task.strip(), deliverable=deliverable, route=decision.route,
                         triage_source=decision.source, trace=self.trace, board=board.snapshot(),
                         cost_usd=self._spent, steps_run=len(plan.steps), errors=errors)

    def _triage(self, task: str, board: Blackboard) -> TriageDecision:
        model = self._cheap_model()
        prompt = "Task: {}\n\nAgents available:\n{}\n\nDecide the route.".format(task.strip(), self.roster.summary())
        try:
            decision, _ = self._ask_json("triage", model, TRIAGE_SYSTEM, prompt,
                                         lambda text: parse_triage(text, self.roster), step_id="triage")
            return decision
        except ContractViolation as exc:
            # A model that cannot follow the contract is a routing failure, not a crash.
            fallback = self._fallback_agent()
            self.trace.record("fallback", phase="triage", reason=str(exc)[:300], agent=fallback)
            return TriageDecision(route="direct", agent=fallback, reason="triage contract failed",
                                  source="fallback")

    def _plan(self, task: str, board: Blackboard) -> Plan:
        model = self._cheap_model()
        prompt = ("Task: {}\n\nAgents available:\n{}\n\nProduce the plan. At most {} steps.".format(
            task.strip(), self.roster.summary(), self.settings.max_steps))
        try:
            plan, _ = self._ask_json("planner", model, PLANNER_SYSTEM, prompt,
                                     lambda text: parse_plan(text, self.roster, self.settings.max_steps),
                                     step_id="plan")
            return plan
        except ContractViolation as exc:
            agent = self._fallback_agent()
            self.trace.record("fallback", phase="plan", reason=str(exc)[:300], agent=agent)
            return Plan(steps=[Step(id="fallback-1", agent=agent, instruction=task.strip(),
                                    writes=(self._default_write(agent),))],
                        rationale="planner contract failed, one step fallback")

    def _execute(self, plan: Plan, board: Blackboard) -> None:
        for index, step in enumerate(plan.steps):
            self._check_budget("step {}".format(step.id))
            if index >= self.settings.max_steps:
                raise BudgetExceeded("step ceiling of {} reached".format(self.settings.max_steps),
                                     trace=self.trace)
            agent = self.roster.get(step.agent)
            available = board.slice(step.reads)
            missing = [r for r in step.reads if r not in available]
            prompt_parts = ["Task: {}".format(board.read("task", "")), "",
                            "Your step: {}".format(step.instruction)]
            if available:
                import json
                prompt_parts += ["", "Shared state you were told to read:",
                                 json.dumps(available, indent=2, sort_keys=True)[:12000]]
            if missing:
                prompt_parts += ["", "Not yet on the board: {}".format(", ".join(missing))]
            prompt_parts += ["", "Write: {}".format(", ".join(step.writes)), STEP_FORMAT]
            self.trace.record("step_start", step_id=step.id, agent=step.agent, model=agent.model,
                              reads=list(step.reads), writes=list(step.writes))
            try:
                completion, _ = self._ask(step.agent, agent.model, agent.system_prompt,
                                          "\n".join(prompt_parts), step_id=step.id)
                if completion.finish_reason == "length":
                    bigger = (self.settings.max_output_tokens * 3
                              if self.settings.max_output_tokens else 3072)
                    self.trace.record("truncated", step_id=step.id, agent=step.agent,
                                      first_budget=self.settings.max_output_tokens,
                                      retry_budget=bigger)
                    completion, _ = self._ask(step.agent, agent.model, agent.system_prompt,
                                              "\n".join(prompt_parts), step_id=step.id,
                                              max_tokens=bigger)
                if not (completion.text or "").strip():
                    # A step that writes nothing must not reach the deliverable as a
                    # blank answer. An earlier live run shipped an empty report this way.
                    self.trace.record("empty_write", step_id=step.id, agent=step.agent,
                                      finish_reason=completion.finish_reason)
                    raise ContractViolation(
                        "step {!r} produced no content (finish_reason={!r})".format(
                            step.id, completion.finish_reason))
            except ProviderError as exc:
                self.trace.record("step_error", step_id=step.id, error=str(exc)[:300])
                raise
            value: Any = completion.text
            try:
                value = extract_json(completion.text)
            except ContractViolation:
                pass                                   # prose is a legitimate artifact
            for key in step.writes:
                board.write(key, value, written_by=step.agent, step_id=step.id)

    def _deliver(self, board: Blackboard, plan: Plan) -> "tuple[Any, str]":
        for candidate in ("deliverable", "report", "summary", "output"):
            if candidate in board.keys():
                return board.read(candidate), candidate
        last_written = next((key for key in reversed(
            [k for e in board.entries() for k in [e.key]])), None)
        if last_written is None:
            raise ConfigError("no step wrote anything, so there is nothing to deliver")
        return board.read(last_written), last_written

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

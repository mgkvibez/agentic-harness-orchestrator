# Changelog

## v0.3.1 - 2026-10-30

Phase 0.1: independent audit remediation. Eight findings, all reproduced first.

- Budget is thread-safe: atomic reserve/settle/release under one lock, typed
  `ReservationError` for duplicate or unknown operations, and open reservations claimed as
  uncertain at the end of a run.
- Cost is reported in four categories (confirmed, estimated, uncertain, outstanding).
  A call that may have been billed without usable usage is never released as free; a
  request rejected before doing work is recorded as not billed.
- Settlement overrun is detected and emitted as `budget_overrun`; exceeding the limit
  exhausts the budget and refuses every later call.
- Malformed usage (negative, boolean, string, NaN) raises `ProviderBadResponse`; absent
  usage is estimated and flagged. Non-JSON bodies with HTTP 200 raise `ProviderBadResponse`
  with a truncated, credential-free snippet.
- Plan validation rejects duplicate producers, duplicate declarations in one step, and
  self-dependencies, so an invalid plan costs no step calls.
- The `allow_overwrite` bypass is removed; corrections go through an authority-gated,
  reason-carrying, audited `replace()`.
- Retry counts are attached to each call rather than read from shared mutable state.
- Every run, including one that dies on an unexpected exception, ends with `run_end`.

## v0.3.0 - 2026-10-30

Phase 0: core reliability, governance and execution integrity. Remediation, not features.

- Pre-call budget reservations covering triage, planning, steps, JSON repair, truncation
  retries and provider retries. A zero budget refuses at the first call.
- Unknown pricing is flagged and quoted conservatively instead of reading as free.
  Estimated and actual cost are recorded separately.
- Write authority derived from the roster and enforced by the blackboard: reserved system
  keys protected, user inputs unable to overwrite system state after seeding, one agent
  unable to replace another's artifact, every refusal audited as `write_rejected`.
- Plan validation: invalid references rejected, cycles rejected by name, undeclared writes
  rejected, and dependency-violating order repaired deterministically with the reordering
  recorded. A step never executes with a missing required input.
- Provider error taxonomy (auth, rate limit, unavailable, timeout, bad response, invalid
  request, model not found), bounded retries that are budgeted before the call, enforced
  timeouts, and transport failures made retryable.
- Evidence: run ids, call ids, artifact sha256 hashes, attempts, validation outcomes,
  terminal status, budget snapshots, and credential scrubbing.
- Multi-output steps now produce distinct artifacts instead of one value copied to every
  declared key.
- CLI exit code 6 for provider failures (previously swallowed as a usage error) and 7 for
  refused execution. Gate extended to assert both.

## v0.2.0 - 2026-10-30

Live-tested against real models, and the real models found real bugs.

- `OpenAICompatibleProvider`: any OpenAI-compatible endpoint (NVIDIA NIM, Groq, Together,
  vLLM, LM Studio) over urllib, zero dependencies, with bounded retries on 429/5xx and a
  fast failure on 4xx. Transport is injectable, so retry and parsing are tested offline.
- Truncation is no longer mistaken for a contract violation. Reasoning models spend the
  output budget thinking, so a truncated answer now buys more room once and records a
  `truncated` event; three of ten real models needed exactly that.
- Steps retried truncation too, and an empty step write now fails loudly. A live run had
  shipped a blank report because the writer hit its cap and nothing checked the content.
- `ModelProfile.rank` breaks ties between unpriced models. Without it, "cheapest" fell
  through to alphabetical order, which sent triage to whichever model sorted first.
- `Orchestrator.triage(task)` answers "how would you route this?" without running steps.
- Scripted replies can be consumed once, so retry paths are expressible in tests.

## v0.1.0 - 2026-10-30

- First release: triage, planner, blackboard, orchestrator loop, execution trace, CLI.
- Strict JSON contracts for triage and plan, with one repair attempt and a logged fallback.
- Three enforced ceilings: steps, cost, wall clock.
- Provider protocol with a scripted offline provider and a lazy-litellm adapter.
- Zero third-party runtime dependencies, verified by installing the wheel into a clean venv.
- Published as `agent-harness-orchestrator` (PyPI rejected shorter variants as too similar to existing projects); the command remains `aharness`.
- 77 tests, no network, no model calls.

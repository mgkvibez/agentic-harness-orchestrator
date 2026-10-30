# Changelog

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

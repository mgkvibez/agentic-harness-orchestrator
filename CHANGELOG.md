# Changelog

## v0.1.0 - 2026-10-30

- First release: triage, planner, blackboard, orchestrator loop, execution trace, CLI.
- Strict JSON contracts for triage and plan, with one repair attempt and a logged fallback.
- Three enforced ceilings: steps, cost, wall clock.
- Provider protocol with a scripted offline provider and a lazy-litellm adapter.
- Zero third-party runtime dependencies, verified by installing the wheel into a clean venv.
- 77 tests, no network, no model calls.

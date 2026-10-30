# agent-harness

**One task in, the right model per sub-task out, with the receipts.** A from-scratch orchestrator: triage decides whether a job needs one agent or several, a planner breaks it into steps, a blackboard carries the artifacts between agents, and every decision lands in an append-only trace with tokens and cost.

[![PyPI](https://img.shields.io/pypi/v/agent-harness-orchestrator.svg)](https://pypi.org/project/agent-harness-orchestrator/)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

Built as the code behind the *Agentic Harness Orchestrator* blueprint: a Python while loop over a shared state dictionary, not a framework wrapper.

![aharness demo: routing each step to a different model, then showing the trace](docs/demo/harness.svg)

```bash
pipx install agent-harness-orchestrator
aharness demo          # offline playback, no keys, no spend
aharness models        # what each model is good at and what it costs
aharness run "Review this codebase for security flaws" --live    # real models
```

Naming: the distribution is `agent-harness-orchestrator`, the import is `agent_harness`, the command is `aharness`. `agent-harness`, `aharness`, `agentharness` and `harness-orchestrator` were either taken or rejected by PyPI as too close to an existing project, so the distribution carries the repository's own name. Nothing you type changes: install the distribution, run `aharness`.

## What the demo actually shows

That capture is a real run of the committed code. Triage and planning went to the free local model, research to the free local model, the judgement step to the reasoning model, and the writing to the cheap one: **$0.00204 for the whole job, 778 tokens in, 299 out**. Send the same four calls to the reasoning model and it costs several times that for no better result on the easy steps.

Honesty about that number: the trace says `provider: fake`, the token counts are estimates (four characters per token) because no model reported usage, and the dollar figure is those estimates against the example price list. It demonstrates the accounting, not a benchmark.

## How it maps to the blueprint

| Blueprint decision | What is in the code |
|---|---|
| Abstraction layer, core never calls a provider API | `providers/base.py` defines `Provider`; `providers/litellm_provider.py` is the only file that imports a vendor SDK, and only when `--live` is passed |
| Simple tasks bypass the multi-agent loop | Triage returns `direct` with one agent, or `decompose`; the direct route skips planning entirely |
| LLM-based routing with a cheap model | Triage and planning both go to `registry.cheapest_with("cheap")`, never the flagship |
| Agent team testing is Phase 2 | Not here. This is the harness only, so it can be stable first |
| Strengths from static config | `examples/models.json`: hand-curated profiles with strengths, speed and price per 1k tokens |
| Blackboard instead of a growing transcript | `blackboard.py`: keyed values, versions, provenance, a per-key and total size ceiling |
| Custom implementation first | No LangGraph, no LangChain. A loop, a dict, and strict JSON contracts |

## The contracts, and what happens when a model ignores them

Every model answer that drives control flow is JSON and is validated before it is trusted: the route must be `direct` or `decompose`, a named agent must exist in the roster, every step must declare the blackboard keys it writes, and the plan cannot exceed the step ceiling.

A reply that fails validation gets **one repair attempt** with the error fed back. A second failure is not a crash: it becomes a logged fallback (`fallback` event in the trace, `triage_source: "fallback"` on the result) and the run continues on the safe single-agent path. You can see which happened without reading logs, because it is a field.

## Bounds

Three ceilings, all enforced per step, all reported with the partial trace when they trip: `--max-steps` (default 8), `--max-cost` (default $0.25), and a wall clock (default 120s). `--max-cost 0` is a valid, useful setting meaning "do not spend", and it means that.

Exit codes are a contract: `0` success, `2` usage or config error, `3` contract violation, `4` budget exceeded, `5` missing optional dependency.

## What this is not

- **Not benchmarked on routing quality.** The routing decision is a model call with a JSON contract. Whether it picks the best agent is unmeasured, and the README does not claim otherwise.
- **Not a tool-calling framework.** Agents read and write blackboard keys. No filesystem, no shell, no browsing.
- **Not a team simulator.** No agent-to-agent debate or critique loops. The blueprint kept that for Phase 2, deliberately.
- **No committed model calls in the suite.** 77 tests run against a scripted provider in under a second, so the suite proves the mechanics, not model quality. `--live` is the real path and needs the `[litellm]` extra and your own keys.
- **Price table is a snapshot**, hand-entered from public pages. It will drift; edit `examples/models.json`.

## Layout

```
agent_harness/
  registry.py       model profiles, cheapest-capable selection
  roster.py         agents: model + system prompt + declared outputs
  plan.py           the two JSON contracts, extraction and validation
  blackboard.py     shared state with provenance and size ceilings
  orchestrator.py   triage, plan, execute, deliver
  trace.py          append-only JSONL plus the table you saw above
  providers/        base protocol, scripted fake, litellm adapter
  cli.py            aharness run / demo / models / agents / trace
```

## License

MIT.

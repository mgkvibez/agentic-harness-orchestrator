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

## Live run against real models (NVIDIA NIM)

The scripted suite proves mechanics. This section is what happened when the harness was pointed at a real OpenAI-compatible endpoint with a real key, because that is the only way to find out whether models honour the contracts.

**The endpoint's model list is not a capability list.** `/v1/models` advertised 85 ids. Of the 58 chat-capable ones, 43 returned `404 Function not found for account` (not provisioned on the key), several hit end-of-life (`410 Gone`, e.g. `meta/llama-3.1-8b-instruct`), and a handful return `200` with empty content when the token budget is smaller than their hidden reasoning. Ten actually answered. `scripts/live_probe.py` exists because of this: probe, do not trust the catalogue.

**Contract compliance across those ten**, asking each to route a task as the harness's own triage step does:

| model | contract | route it chose | latency |
|---|---|---|---|
| nemotron-3.5-bolt-e4b | clean | direct | 0.8s |
| nemotron-3-nano-omni-30b-a3b-reasoning | clean | direct | 2.0s |
| nemotron-3-ultra-550b-a55b | clean | decompose | 3.5s |
| ising-calibration-1.5-31b | clean | decompose | 2.8s |
| gpt-oss-20b | clean | direct | 6.9s |
| minimax-m3.1-flash | clean | direct | 12.1s |
| nemotron-3.5-lightning-30b-a3b | clean after buying room | direct | 16.9s |
| muse-glimmer-30b | clean after buying room | decompose | 43.2s |
| mistral-large-4-1t-a52b | clean after buying room | decompose | 62.3s |
| laguna-xs-2.1 | provider error | - | 1.7s |

Nine of ten complied. The three that needed help are the interesting part: they are reasoning models that spend the output budget thinking, so a 320-token ceiling truncated the answer and looked identical to a model ignoring the contract. The harness now distinguishes the two, buys more room once, and records `truncated` in the trace.

**A full orchestrated run**, real models, real tokens, no script: triage and planning on the smallest ranked model, analysis on `gpt-oss-20b`, writing on `nemotron-3.5-bolt-e4b`.

```
STEP     AGENT        MODEL                       IN     OUT COST(USD)       MS
-------------------------------------------------------------------------------
triage   triage       nvidia/nemotron-3.5-bo     159      41   0.00000      952
plan     planner      nvidia/nemotron-3.5-bo     199     174   0.00000     3157
s1       analyst      openai/gpt-oss-20b         246     275   0.00000     8214
s2       researcher   nvidia/nemotron-3.5-li     310    1500   0.00000    14221
s2       researcher   nvidia/nemotron-3.5-li     310    2528   0.00000    25796
s3       writer       nvidia/nemotron-3.5-bo     661     345   0.00000     5816
-------------------------------------------------------------------------------
totals: 1885 in / 4863 out tokens, $0.00000, 6 step(s)
```

58 seconds wall clock, six provider calls for five logical steps, one of them the truncation retry on `s2`. The deliverable correctly named the mounted docker socket, the `curl | bash`, and root SSH login, with evidence per finding. The dollars read zero because NIM is credit-metered and the profiles carry no per-token price: the token counts are measured, the cost column is not.

Two honest observations from that run, both visible in the trace:

- The planner ordered `analyst` before `researcher`, so the analyst ran with its declared input `findings` still absent. The harness records the missing key in the prompt and lets the step proceed. That is the wrong default for a plan that declares dependencies, and it is the next thing to fix.
- Reasoning models can burn a 1500-token budget before producing a character, which is why the retry doubles to 4500 rather than nudging.

## What this is not

- **Live-tested on ten NVIDIA NIM models, not benchmarked.** Contract compliance was measured there; whether the routing choice is the best one remains unmeasured, and the harness does not claim otherwise. Routing quality is The routing decision is a model call with a JSON contract. Whether it picks the best agent is unmeasured, and the README does not claim otherwise.
- **Not a tool-calling framework.** Agents read and write blackboard keys. No filesystem, no shell, no browsing.
- **Not a team simulator.** No agent-to-agent debate or critique loops. The blueprint kept that for Phase 2, deliberately.
- **No committed model calls in the suite.** 92 tests run against a scripted provider in under a second, so the suite proves the mechanics, not model quality. `--live` is the real path and needs the `[litellm]` extra and your own keys.
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

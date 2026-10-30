"""Ask every reachable model to honour the harness's own triage contract.

This is the gap the scripted suite cannot close: real models wrap JSON in prose,
refuse formats and time out. One call per model, small token budget, real numbers.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_harness.errors import ProviderError
from agent_harness.config import Settings
from agent_harness.orchestrator import Orchestrator
from agent_harness.plan import parse_triage
from agent_harness.providers.base import Message
from agent_harness.providers.openai_compat import OpenAICompatibleProvider
from agent_harness.registry import load_profiles
from agent_harness.roster import Agent, Roster

ROSTER = Roster([
    Agent("researcher", "researcher", "m", "read", ("findings",)),
    Agent("analyst", "analyst", "m", "judge", ("issues",)),
    Agent("writer", "writer", "m", "write", ("deliverable",)),
], registry=None)

PROMPT = ("Task: Review this codebase for security flaws.\n\n"
          "Agents available:\n{}\n\nDecide the route.").format(ROSTER.summary())


def main() -> int:
    base_url = os.environ.get("AHARNESS_BASE_URL", "https://integrate.api.nvidia.com/v1")
    provider = OpenAICompatibleProvider(base_url, os.environ["NVIDIA_API_KEY"])
    from agent_harness.registry import ModelRegistry
    profiles = load_profiles(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "agent_harness", "examples", "nvidia-models.json"))
    rows = []
    for profile in profiles:
        started = time.monotonic()
        record = {"model": profile.id.split("/")[-1][:34], "model_id": profile.id}
        import dataclasses
        triage_profile = dataclasses.replace(
            profile, strengths=tuple(sorted(set(profile.strengths) | {"cheap"})))
        one = ModelRegistry([triage_profile])
        orchestrator = Orchestrator(one, ROSTER, provider, Settings(max_output_tokens=320))
        try:
            decision = orchestrator.triage("Review this codebase for security flaws")
            events = orchestrator.trace.events
            spends = [e for e in events if e.kind == "step_end"]
            record.update(ok=True, contract="clean", route=decision.route, agent=decision.agent,
                          input_tokens=sum(e.fields.get("input_tokens", 0) for e in spends),
                          output_tokens=sum(e.fields.get("output_tokens", 0) for e in spends),
                          retried=bool(orchestrator.trace.of_kind("truncated")),
                          source=decision.source)
        except Exception as exc:
            record.update(ok=False, contract="error", error=str(exc)[:90])
        record["ms"] = int((time.monotonic() - started) * 1000)
        rows.append(record)
        print(json.dumps(record))
    clean = sum(1 for r in rows if r.get("contract") == "clean")
    print("\n=== summary: {}/{} models honoured the contract first try".format(clean, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

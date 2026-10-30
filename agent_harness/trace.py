"""The receipt: every decision, model, token and dollar, in order.

Append-only JSONL on disk plus a human summary. If a run cannot be explained
afterwards, the harness failed, not the model.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Event:
    kind: str
    fields: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def as_dict(self) -> Dict[str, Any]:
        return {"ts": round(self.ts, 3), "kind": self.kind, **self.fields}


class Trace:
    def __init__(self, path: str = ""):
        self.path = path
        self.events: List[Event] = []

    def record(self, kind: str, **fields) -> Event:
        event = Event(kind=kind, fields=fields)
        self.events.append(event)
        if self.path:
            self._append(event)
        return event

    def _append(self, event: Event) -> None:
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event.as_dict(), sort_keys=True) + "\n")

    # totals -----------------------------------------------------------------
    def total_cost(self) -> float:
        # Only step_end events are spend events. run_end also carries a cost field
        # for humans, and summing every event double-counted every run.
        return round(sum(float(e.fields.get("cost_usd", 0.0))
                         for e in self.events if e.kind == "step_end"), 6)

    def total_tokens(self) -> Dict[str, int]:
        spends = [e for e in self.events if e.kind == "step_end"]
        return {"input": sum(int(e.fields.get("input_tokens", 0)) for e in spends),
                "output": sum(int(e.fields.get("output_tokens", 0)) for e in spends)}

    def of_kind(self, kind: str) -> List[Event]:
        return [e for e in self.events if e.kind == kind]

    def summary(self) -> str:
        lines = []
        header = "{:<8} {:<12} {:<22} {:>7} {:>7} {:>9} {:>8}".format(
            "STEP", "AGENT", "MODEL", "IN", "OUT", "COST(USD)", "MS")
        lines.append(header)
        lines.append("-" * len(header))
        for event in self.events:
            if event.kind != "step_end":
                continue
            f = event.fields
            lines.append("{:<8} {:<12} {:<22} {:>7} {:>7} {:>9} {:>8}".format(
                str(f.get("step_id", ""))[:8], str(f.get("agent", ""))[:12],
                str(f.get("model", ""))[:22], f.get("input_tokens", 0), f.get("output_tokens", 0),
                "{:.5f}".format(float(f.get("cost_usd", 0.0))), f.get("duration_ms", 0)))
        tokens = self.total_tokens()
        lines.append("-" * len(header))
        lines.append("totals: {} in / {} out tokens, ${:.5f}, {} step(s)".format(
            tokens["input"], tokens["output"], self.total_cost(), len(self.of_kind("step_end"))))
        return "\n".join(lines)

    def to_jsonl(self, path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(directory, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=directory, delete=False, encoding="utf-8") as fh:
            for event in self.events:
                fh.write(json.dumps(event.as_dict(), sort_keys=True) + "\n")
            tmp = fh.name
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "Trace":
        trace = cls()
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                kind = payload.pop("kind", "unknown")
                payload.pop("ts", None)
                trace.events.append(Event(kind=kind, fields=payload))
        return trace

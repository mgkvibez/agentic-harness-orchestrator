"""The receipt: every decision, call, artifact and dollar, in order.

Evidence is only useful if it is complete and safe. Every event carries the run
id, every provider call carries a call id, every artifact carries a content hash,
estimated and actual costs are recorded separately, and strings are scrubbed of
anything that looks like a credential before they are written.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Anything shaped like a credential is redacted on the way in. The trace is
# written to disk and pasted into reports, so it must never carry a key.
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"nvapi-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"ghp_[A-Za-z0-9]{8,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{6,}"),
    re.compile(r"(?i)(api[_-]?key\"?\s*[:=]\s*)\"?[A-Za-z0-9._\-]{6,}"),
    re.compile(r"(?i)(authorization\"?\s*[:=]\s*)\"?[^\s,\"]{6,}"),
)

REDACTED = "[redacted]"


def scrub(value: Any) -> Any:
    """Recursively remove credential-shaped strings from anything logged."""
    if isinstance(value, str):
        cleaned = value
        for pattern in _SECRET_PATTERNS:
            cleaned = pattern.sub(
                lambda m: (m.group(1) + REDACTED) if m.groups() else REDACTED, cleaned)
        return cleaned
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(v) for v in value]
    return value


@dataclass
class Event:
    kind: str
    fields: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def as_dict(self) -> Dict[str, Any]:
        return {"ts": round(self.ts, 3), "kind": self.kind, **self.fields}


class Trace:
    def __init__(self, path: str = "", run_id: str = ""):
        self.path = path
        self.run_id = run_id
        self.events: List[Event] = []

    def record(self, kind: str, **fields) -> Event:
        payload = scrub(fields)
        if self.run_id:
            payload.setdefault("run_id", self.run_id)
        event = Event(kind=kind, fields=payload)
        self.events.append(event)
        if self.path:
            self._append(event)
        return event

    def _append(self, event: Event) -> None:
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event.as_dict(), sort_keys=True) + "\n")

    # -- totals --------------------------------------------------------------
    def total_cost(self) -> float:
        """Actual spend only. run_end carries a cost for humans, not for maths."""
        return round(sum(float(e.fields.get("cost_usd", 0.0))
                         for e in self.events if e.kind == "step_end"), 6)

    def estimated_cost(self) -> float:
        return round(sum(float(e.fields.get("estimated_cost_usd", 0.0))
                         for e in self.events if e.kind == "step_end"), 6)

    def total_tokens(self) -> Dict[str, int]:
        spends = [e for e in self.events if e.kind == "step_end"]
        return {"input": sum(int(e.fields.get("input_tokens", 0)) for e in spends),
                "output": sum(int(e.fields.get("output_tokens", 0)) for e in spends)}

    def of_kind(self, kind: str) -> List[Event]:
        return [e for e in self.events if e.kind == kind]

    def artifacts(self) -> List[Dict[str, Any]]:
        return [e.fields for e in self.of_kind("artifact_written")]

    def summary(self) -> str:
        lines = []
        header = "{:<10} {:<12} {:<22} {:>7} {:>7} {:>10} {:>5} {:>8}".format(
            "STEP", "AGENT", "MODEL", "IN", "OUT", "COST(USD)", "TRY", "MS")
        lines.append(header)
        lines.append("-" * len(header))
        for event in self.events:
            if event.kind != "step_end":
                continue
            f = event.fields
            lines.append("{:<10} {:<12} {:<22} {:>7} {:>7} {:>10} {:>5} {:>8}".format(
                str(f.get("step_id", ""))[:10], str(f.get("agent", ""))[:12],
                str(f.get("model", ""))[:22], f.get("input_tokens", 0), f.get("output_tokens", 0),
                "{:.5f}".format(float(f.get("cost_usd", 0.0))),
                f.get("attempts", 1), f.get("duration_ms", 0)))
        tokens = self.total_tokens()
        lines.append("-" * len(header))
        lines.append("totals: {} in / {} out tokens, ${:.5f} actual / ${:.5f} estimated, "
                     "{} call(s), {} artifact(s)".format(
                         tokens["input"], tokens["output"], self.total_cost(),
                         self.estimated_cost(), len(self.of_kind("step_end")),
                         len(self.artifacts())))
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

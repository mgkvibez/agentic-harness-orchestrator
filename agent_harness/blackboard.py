"""Shared state: a keyed board agents read and write, with provenance.

Not a chat transcript. Agents get the slices they need by name, which is the
whole point: context stays small and the run stays auditable.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .errors import HarnessError


@dataclass(frozen=True)
class Entry:
    key: str
    value: Any
    written_by: str
    step_id: str
    version: int
    chars: int

    def as_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "written_by": self.written_by, "step_id": self.step_id,
                "version": self.version, "chars": self.chars}


class Blackboard:
    def __init__(self, char_limit: int = 40000):
        self.char_limit = char_limit
        self._values: Dict[str, Any] = {}
        self._entries: List[Entry] = []
        self._versions: Dict[str, int] = {}
        self._history: Dict[str, List[Any]] = {}

    # writes -----------------------------------------------------------------
    def write(self, key: str, value: Any, *, written_by: str = "", step_id: str = "") -> Entry:
        if not key or not isinstance(key, str):
            raise HarnessError("blackboard keys must be non-empty strings")
        # Measure the value being written, not the one already there: the earlier
        # version sized the existing value, so every limit check validated stale data.
        try:
            size = len(json.dumps(value, sort_keys=True))
        except (TypeError, ValueError):
            raise HarnessError("blackboard key {!r} is not JSON-serialisable".format(key)) from None
        if size > self.char_limit:
            raise HarnessError("write of {!r} is {} chars, over the {}-char board limit".format(
                key, size, self.char_limit))
        if self.total_chars() - self._chars_for(key) + size > self.char_limit:
            raise HarnessError("board would exceed its {}-char limit; trim or consolidate keys".format(
                self.char_limit))
        version = self._versions.get(key, 0) + 1
        self._versions[key] = version
        self._history.setdefault(key, []).append(self._values.get(key))
        self._values[key] = value
        entry = Entry(key=key, value=value, written_by=written_by, step_id=step_id,
                      version=version, chars=size)
        self._entries.append(entry)
        return entry

    def read(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self._values:
            raise HarnessError("step needed {!r} but nothing has written it yet".format(key))
        return self._values[key]

    def keys(self) -> List[str]:
        return sorted(self._values)

    def entries(self) -> List[Entry]:
        return list(self._entries)

    def slice(self, keys: Iterable[str]) -> Dict[str, Any]:
        return {k: self._values[k] for k in keys if k in self._values}

    def history(self, key: str) -> List[Any]:
        return list(self._history.get(key, []))

    def snapshot(self) -> Dict[str, Any]:
        return dict(self._values)

    def total_chars(self) -> int:
        return sum(self._chars_for(k) for k in self._values)

    def save(self, path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(directory, exist_ok=True)
        payload = {"values": self._values,
                   "entries": [e.as_dict() for e in self._entries]}
        with tempfile.NamedTemporaryFile("w", dir=directory, delete=False) as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            tmp = fh.name
        os.replace(tmp, path)

    # internals --------------------------------------------------------------
    def _chars_for(self, key: str) -> int:
        if key not in self._values:
            return 0
        try:
            return len(json.dumps(self._values[key], sort_keys=True))
        except (TypeError, ValueError):
            return 0

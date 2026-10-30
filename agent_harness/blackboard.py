"""Shared state: a keyed board with ownership, provenance and enforcement.

Enforcement lives here rather than in the orchestrator because there is no way to
write without passing through this class. An agent may write only the keys its
roster entry declares, reserved system keys belong to the harness, and one agent
cannot silently replace another's artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .errors import HarnessError, OverwriteRejected, PermissionDenied
from .permissions import WritePolicy, is_reserved

ACTOR_AGENT, ACTOR_SYSTEM, ACTOR_USER = "agent", "system", "user"


def artifact_hash(value: Any) -> str:
    """Stable sha256 of the stored value, so a trace can prove what was delivered."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Entry:
    key: str
    value: Any
    written_by: str
    step_id: str
    version: int
    chars: int
    actor: str = ACTOR_AGENT
    sha256: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "written_by": self.written_by, "step_id": self.step_id,
                "version": self.version, "chars": self.chars, "actor": self.actor,
                "sha256": self.sha256}


class Blackboard:
    def __init__(self, char_limit: int = 40000, policy: Optional[WritePolicy] = None):
        self.char_limit = char_limit
        self.policy = policy
        self._values: Dict[str, Any] = {}
        self._owners: Dict[str, str] = {}
        self._entries: List[Entry] = []
        self._versions: Dict[str, int] = {}
        self._history: Dict[str, List[Any]] = {}
        self._seeding = True          # only the harness may seed, and only first

    # -- seeding -------------------------------------------------------------
    def close_seeding(self) -> None:
        """After this, system and user writes can no longer change the board."""
        self._seeding = False

    @property
    def seeding(self) -> bool:
        return self._seeding

    # -- writes --------------------------------------------------------------
    def write(self, key: str, value: Any, *, written_by: str = "", step_id: str = "",
              actor: str = ACTOR_AGENT, allow_overwrite: bool = False) -> Entry:
        if not key or not isinstance(key, str):
            raise HarnessError("blackboard keys must be non-empty strings")
        if actor not in (ACTOR_AGENT, ACTOR_SYSTEM, ACTOR_USER):
            raise HarnessError("unknown actor {!r}".format(actor))

        # 1. authority
        if actor == ACTOR_AGENT:
            if self.policy is not None:
                self.policy.check(written_by, key)      # raises PermissionDenied
            elif is_reserved(key):
                raise PermissionDenied(
                    "reserved key {!r} cannot be written by an agent".format(key),
                    agent=written_by, key=key, reason="reserved key")
        else:
            if is_reserved(key) and actor != ACTOR_SYSTEM:
                raise PermissionDenied(
                    "{} may not write reserved key {!r}".format(actor, key),
                    agent=actor, key=key, reason="reserved key")
            if not self._seeding:
                raise PermissionDenied(
                    "{} writes are closed once the run has started (key {!r})".format(actor, key),
                    agent=actor, key=key, reason="seeding closed")

        # 2. no silent replacement of another actor's artifact. Enforcement is
        #    active whenever a policy is attached, which is every orchestrated
        #    run; a bare board stays a plain container for library use.
        if key in self._values and not allow_overwrite and self.policy is not None:
            owner, owner_actor = self._owners.get(key, ""), self._entries_actor(key)
            same_actor = (actor == ACTOR_AGENT and owner_actor == ACTOR_AGENT and owner == written_by)
            shared = actor in (ACTOR_SYSTEM,) and self._seeding
            if not same_actor and not shared:
                raise OverwriteRejected(
                    "key {!r} already holds an artifact from {!r} ({}); {!r} may not replace it "
                    "without allow_overwrite".format(key, owner or "unknown", owner_actor or "?",
                                                     written_by or actor),
                    key=key, owner=owner, actor=written_by or actor)

        # 3. size and serialisability
        try:
            size = len(json.dumps(value, sort_keys=True, separators=(",", ":")))
        except (TypeError, ValueError):
            raise HarnessError("blackboard key {!r} is not JSON-serialisable".format(key)) from None
        if size > self.char_limit:
            raise HarnessError("write of {!r} is {} chars, over the {}-char board limit".format(
                key, size, self.char_limit))
        if self.total_chars() - self._chars_for(key) + size > self.char_limit:
            raise HarnessError("board would exceed its {}-char limit; trim or consolidate keys"
                               .format(self.char_limit))

        version = self._versions.get(key, 0) + 1
        self._versions[key] = version
        self._history.setdefault(key, []).append(self._values.get(key))
        self._values[key] = value
        self._owners[key] = written_by if actor == ACTOR_AGENT else written_by or actor
        entry = Entry(key=key, value=value, written_by=written_by, step_id=step_id,
                      version=version, chars=size, actor=actor, sha256=artifact_hash(value))
        self._entries.append(entry)
        return entry

    def _entries_actor(self, key: str) -> str:
        for entry in reversed(self._entries):
            if entry.key == key:
                return entry.actor
        return ""

    # -- reads ---------------------------------------------------------------
    def read(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self._values:
            raise HarnessError("step needed {!r} but nothing has written it yet".format(key))
        return self._values[key]

    def has(self, key: str) -> bool:
        return key in self._values

    def missing(self, keys: Iterable[str]) -> List[str]:
        return [k for k in keys if k not in self._values]

    def keys(self) -> List[str]:
        return sorted(self._values)

    def owner(self, key: str) -> str:
        return self._owners.get(key, "")

    def entries(self) -> List[Entry]:
        return list(self._entries)

    def slice(self, keys: Iterable[str]) -> Dict[str, Any]:
        return {k: self._values[k] for k in keys if k in self._values}

    def history(self, key: str) -> List[Any]:
        return list(self._history.get(key, []))

    def snapshot(self) -> Dict[str, Any]:
        return dict(self._values)

    def provenance(self) -> Dict[str, Dict[str, Any]]:
        """key -> who wrote the current version, and its hash."""
        out: Dict[str, Dict[str, Any]] = {}
        for entry in self._entries:
            out[entry.key] = {"written_by": entry.written_by, "step_id": entry.step_id,
                              "version": entry.version, "sha256": entry.sha256,
                              "actor": entry.actor}
        return out

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

    # -- internals -----------------------------------------------------------
    def _chars_for(self, key: str) -> int:
        if key not in self._values:
            return 0
        try:
            return len(json.dumps(self._values[key], sort_keys=True, separators=(",", ":")))
        except (TypeError, ValueError):
            return 0

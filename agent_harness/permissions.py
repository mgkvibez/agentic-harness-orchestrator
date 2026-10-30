"""Who may write what. Enforced by the blackboard, not by convention.

The roster already declares what each agent produces. This module turns that
declaration into a rule the runtime applies, and draws a hard line around the
keys the harness itself owns: a planner cannot hand an agent authority it does
not have, because authority comes from the roster, never from the plan.
"""

from __future__ import annotations

from typing import FrozenSet, Iterable, Optional

from .errors import PermissionDenied
from .roster import Roster

# Reserved keys belong to the harness. Note what is NOT here: user-supplied
# inputs such as "codebase" are user state, not system state, so a run may seed
# them. What matters is that only system writes can touch the reserved set, and
# only before execution begins.
RESERVED_KEYS: FrozenSet[str] = frozenset({"task"})

RESERVED_PREFIX = "_"


def is_reserved(key: str) -> bool:
    return key in RESERVED_KEYS or key.startswith(RESERVED_PREFIX)


class WritePolicy:
    """Agent name -> the keys that agent may write."""

    def __init__(self, roster: Roster, reserved: Iterable[str] = RESERVED_KEYS):
        self._allowed = {agent.name: frozenset(agent.writes) for agent in roster}
        self._reserved = frozenset(reserved)
        # A rogue roster that grants a system key to an agent is a config error,
        # not a permission to be honoured.
        for name, keys in self._allowed.items():
            clash = sorted(k for k in keys if is_reserved(k))
            if clash:
                raise PermissionDenied(
                    "agent {!r} declares reserved system key(s) {}; reserved keys belong to the "
                    "harness".format(name, clash), agent=name, key=clash[0],
                    reason="reserved key in roster")

    def allowed_keys(self, agent: str) -> FrozenSet[str]:
        return self._allowed.get(agent, frozenset())

    def check(self, agent: str, key: str) -> None:
        if agent not in self._allowed:
            raise PermissionDenied(
                "unknown agent {!r} cannot write {!r}".format(agent, key),
                agent=agent, key=key, reason="unknown agent")
        if is_reserved(key):
            raise PermissionDenied(
                "agent {!r} cannot write reserved system key {!r}".format(agent, key),
                agent=agent, key=key, reason="reserved key")
        if key not in self._allowed[agent]:
            raise PermissionDenied(
                "agent {!r} is not authorised to write {!r}; it declares {}".format(
                    agent, key, sorted(self._allowed[agent]) or "nothing"),
                agent=agent, key=key, reason="not declared")

    def can_write(self, agent: Optional[str], key: str) -> bool:
        try:
            self.check(agent or "", key)
            return True
        except PermissionDenied:
            return False

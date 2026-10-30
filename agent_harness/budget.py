"""Pre-call budget reservation with thread-safe, atomic accounting.

Three properties this file exists to guarantee:

1. A reservation either fits or it does not. Check-and-insert happens under one
   lock, so concurrent callers cannot both observe the same headroom.
2. A call that may have been billed is never released as though it were free.
   When usage is unavailable the quote moves to `uncertain_usd`, which counts
   against the limit exactly like real spend but is reported separately and is
   never described as confirmed.
3. Settling more than was reserved is recorded as an overrun rather than quietly
   absorbed, and an exhausted budget refuses every subsequent call.

Every state change happens under `_lock`; nothing outside this class touches the
counters, so a snapshot is always internally consistent.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

from .errors import BudgetExceeded, ReservationError
from .providers.base import estimate_tokens
from .registry import ModelProfile

CONFIRMED = "confirmed"
UNCERTAIN = "uncertain"
RELEASED = "released"


@dataclass(frozen=True)
class Quote:
    """What one logical call is expected to cost, and how confident that is."""
    usd: float
    conservative: bool
    input_tokens: int
    output_tokens: int
    attempts: int

    @property
    def priced(self) -> bool:
        return self.usd > 0


@dataclass(frozen=True)
class Settlement:
    """The outcome of closing a reservation. Never guesses what it cannot know."""
    reservation_id: str
    quoted_usd: float
    actual_usd: float
    overrun_usd: float
    exhausted: bool
    kind: str = CONFIRMED
    reason: str = ""

    @property
    def uncertain(self) -> bool:
        return self.kind == UNCERTAIN


class _Reservation:
    __slots__ = ("id", "usd", "label", "conservative", "state")

    def __init__(self, ident: str, usd: float, label: str, conservative: bool):
        self.id = ident
        self.usd = usd
        self.label = label
        self.conservative = conservative
        self.state = "reserved"


class Budget:
    def __init__(self, limit_usd: Optional[float], unknown_pricing_per_1k: float = 0.01):
        self.limit_usd = limit_usd
        self.unknown_pricing_per_1k = unknown_pricing_per_1k
        self._lock = threading.RLock()
        self._spent_usd = 0.0          # confirmed actual expenditure
        self._uncertain_usd = 0.0      # may have been billed; usage unavailable
        self._estimated_usd = 0.0      # sum of settled quotes, for reporting only
        self._overrun_usd = 0.0        # amount settled beyond its reservation
        self._open: Dict[str, _Reservation] = {}
        self._closed: Dict[str, _Reservation] = {}
        self._exhausted_reason = ""

    # -- quoting (pure, no state) -------------------------------------------
    def quote(self, profile: ModelProfile, prompt_tokens: int, max_output_tokens: int,
              attempts: int = 1) -> Quote:
        per_attempt_usd, conservative = profile.estimate_cost(
            prompt_tokens, max_output_tokens, self.unknown_pricing_per_1k)
        attempts = max(1, int(attempts))
        return Quote(usd=round(per_attempt_usd * attempts, 6), conservative=conservative,
                     input_tokens=prompt_tokens * attempts,
                     output_tokens=max_output_tokens * attempts, attempts=attempts)

    def quote_for_prompt(self, profile: ModelProfile, prompt: str, max_output_tokens: int,
                         attempts: int = 1) -> Quote:
        return self.quote(profile, estimate_tokens(prompt), max_output_tokens, attempts)

    # -- view ----------------------------------------------------------------
    @property
    def spent_usd(self) -> float:
        with self._lock:
            return round(self._spent_usd, 6)

    @property
    def uncertain_usd(self) -> float:
        with self._lock:
            return round(self._uncertain_usd, 6)

    @property
    def committed_usd(self) -> float:
        with self._lock:
            return round(self._spent_usd + self._uncertain_usd, 6)

    @property
    def estimated_usd(self) -> float:
        with self._lock:
            return round(self._estimated_usd, 6)

    @property
    def overrun_usd(self) -> float:
        with self._lock:
            return round(self._overrun_usd, 6)

    @property
    def reserved_usd(self) -> float:
        with self._lock:
            return round(sum(r.usd for r in self._open.values()), 6)

    @property
    def open_reservations(self) -> List[str]:
        with self._lock:
            return sorted(self._open)

    @property
    def exhausted_reason(self) -> str:
        with self._lock:
            return self._exhausted_reason

    @property
    def remaining_usd(self) -> Optional[float]:
        with self._lock:
            if self.limit_usd is None:
                return None
            return round(self.limit_usd - self._spent_usd - self._uncertain_usd
                         - sum(r.usd for r in self._open.values()), 6)

    # -- reservations --------------------------------------------------------
    def reserve(self, quote: Quote, label: str = "") -> str:
        """Take the money out before the call, atomically."""
        with self._lock:
            if self._exhausted_reason:
                raise BudgetExceeded(
                    "budget is exhausted ({}); refusing {}".format(
                        self._exhausted_reason, label or "the call"), quote=quote)
            if self.limit_usd is not None:
                projected = round(self._spent_usd + self._uncertain_usd
                                  + sum(r.usd for r in self._open.values()) + quote.usd, 6)
                if projected > self.limit_usd + 1e-12:
                    raise BudgetExceeded(
                        "budget would be exceeded before {}: quote ${:.6f}, confirmed ${:.6f}, "
                        "uncertain ${:.6f}, reserved ${:.6f}, limit ${:.6f}{}".format(
                            label or "the call", quote.usd, self._spent_usd, self._uncertain_usd,
                            sum(r.usd for r in self._open.values()), self.limit_usd,
                            " (quote uses a conservative rate for unknown pricing)"
                            if quote.conservative else ""),
                        quote=quote)
            reservation = _Reservation(uuid.uuid4().hex[:10], quote.usd, label,
                                       quote.conservative)
            self._open[reservation.id] = reservation
            return reservation.id

    def _take(self, reservation_id: str) -> _Reservation:
        """Remove a reservation from the open set, refusing duplicates."""
        reservation = self._open.pop(reservation_id, None)
        if reservation is not None:
            self._closed[reservation_id] = reservation
            reservation.state = "closed"
            return reservation
        previous = self._closed.get(reservation_id)
        if previous is None:
            raise ReservationError(
                "reservation {!r} is unknown; it was never created by this budget"
                .format(reservation_id))
        raise ReservationError(
            "reservation {!r} was already {}; a reservation can only be closed once"
            .format(reservation_id, previous.state))
        # note: state is set below for clarity of message

    def settle(self, reservation_id: str, actual_usd: float) -> Settlement:
        """Convert a reservation into confirmed spend and report any overrun."""
        with self._lock:
            reservation = self._take(reservation_id)
            actual = round(max(0.0, float(actual_usd)), 6)
            self._spent_usd = round(self._spent_usd + actual, 6)
            self._estimated_usd = round(self._estimated_usd + reservation.usd, 6)
            overrun = round(max(0.0, actual - reservation.usd), 6)
            if overrun:
                self._overrun_usd = round(self._overrun_usd + overrun, 6)
            exhausted = self._mark_exhausted()
            return Settlement(reservation_id=reservation_id, quoted_usd=reservation.usd,
                              actual_usd=actual, overrun_usd=overrun, exhausted=exhausted,
                              kind=CONFIRMED)

    def settle_uncertain(self, reservation_id: str, reason: str) -> Settlement:
        """A call that may have been billed, with no usable usage.

        The quote moves into `uncertain_usd`: it consumes headroom like real
        spend, it is reported as unconfirmed, and it is never described as free.
        """
        with self._lock:
            reservation = self._take(reservation_id)
            self._uncertain_usd = round(self._uncertain_usd + reservation.usd, 6)
            self._estimated_usd = round(self._estimated_usd + reservation.usd, 6)
            exhausted = self._mark_exhausted()
            return Settlement(reservation_id=reservation_id, quoted_usd=reservation.usd,
                              actual_usd=0.0, overrun_usd=0.0, exhausted=exhausted,
                              kind=UNCERTAIN, reason=reason[:200])

    def release(self, reservation_id: str) -> Settlement:
        """Give back a reservation for a call that provably did not cost anything."""
        with self._lock:
            reservation = self._take(reservation_id)
            reservation.state = RELEASED
            return Settlement(reservation_id=reservation_id, quoted_usd=reservation.usd,
                              actual_usd=0.0, overrun_usd=0.0,
                              exhausted=bool(self._exhausted_reason), kind=RELEASED)

    def _mark_exhausted(self) -> bool:
        if self.limit_usd is None:
            return False
        committed = self._spent_usd + self._uncertain_usd
        if committed > self.limit_usd + 1e-12 and not self._exhausted_reason:
            self._exhausted_reason = (
                "confirmed ${:.6f} plus uncertain ${:.6f} exceeded the ${:.6f} limit"
                .format(self._spent_usd, self._uncertain_usd, self.limit_usd))
            return True
        return bool(self._exhausted_reason)

    def claim_leaks(self) -> List[Settlement]:
        """Close any reservation left open, counting it as uncertain, not free."""
        with self._lock:
            leaked = list(self._open)
        return [self.settle_uncertain(rid, "reservation left open at end of run")
                for rid in leaked]

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            committed = round(self._spent_usd + self._uncertain_usd, 6)
            remaining = (None if self.limit_usd is None else
                         round(self.limit_usd - committed
                               - sum(r.usd for r in self._open.values()), 6))
            return {"limit_usd": self.limit_usd,
                    "spent_usd": round(self._spent_usd, 6),
                    "uncertain_usd": round(self._uncertain_usd, 6),
                    "committed_usd": committed,
                    "estimated_usd": round(self._estimated_usd, 6),
                    "overrun_usd": round(self._overrun_usd, 6),
                    "reserved_usd": round(sum(r.usd for r in self._open.values()), 6),
                    "remaining_usd": remaining,
                    "open_reservations": len(self._open),
                    "exhausted_reason": self._exhausted_reason}

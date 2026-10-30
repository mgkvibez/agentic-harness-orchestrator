"""Pre-call budget reservation, so a call cannot happen unless it is paid for.

The rule this enforces: every provider call is quoted, reserved and settled
before the next one is allowed. Nothing is spent on the strength of a check that
happens afterwards, which is what the previous implementation did.

Unknown pricing is never treated as free. A model whose price is not modelled
(credit-metered endpoints, private deployments) is quoted at a configurable
conservative rate and flagged, so a zero budget still blocks it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

from .errors import BudgetExceeded
from .providers.base import estimate_tokens
from .registry import ModelProfile


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


@dataclass
class _Reservation:
    id: str
    usd: float
    label: str
    conservative: bool


@dataclass
class Budget:
    limit_usd: Optional[float]
    unknown_pricing_per_1k: float = 0.01
    spent_usd: float = 0.0
    estimated_usd: float = 0.0
    _open: Dict[str, _Reservation] = field(default_factory=dict)

    # -- quoting -------------------------------------------------------------
    def quote(self, profile: ModelProfile, prompt_tokens: int, max_output_tokens: int,
              attempts: int = 1) -> Quote:
        """Worst case for the whole logical call, including permitted retries."""
        per_attempt_usd, conservative = profile.estimate_cost(
            prompt_tokens, max_output_tokens, self.unknown_pricing_per_1k)
        attempts = max(1, int(attempts))
        return Quote(usd=round(per_attempt_usd * attempts, 6), conservative=conservative,
                     input_tokens=prompt_tokens * attempts,
                     output_tokens=max_output_tokens * attempts, attempts=attempts)

    def quote_for_prompt(self, profile: ModelProfile, prompt: str, max_output_tokens: int,
                         attempts: int = 1) -> Quote:
        return self.quote(profile, estimate_tokens(prompt), max_output_tokens, attempts)

    # -- reservations --------------------------------------------------------
    @property
    def reserved_usd(self) -> float:
        return round(sum(r.usd for r in self._open.values()), 6)

    @property
    def remaining_usd(self) -> Optional[float]:
        if self.limit_usd is None:
            return None
        return round(self.limit_usd - self.spent_usd - self.reserved_usd, 6)

    def reserve(self, quote: Quote, label: str = "") -> str:
        """Take the money out before the call. Refuses if it is not there."""
        if self.limit_usd is not None:
            projected = round(self.spent_usd + self.reserved_usd + quote.usd, 6)
            if projected > self.limit_usd + 1e-12:
                raise BudgetExceeded(
                    "budget would be exceeded before {}: quote ${:.6f}, already spent ${:.6f}, "
                    "reserved ${:.6f}, limit ${:.6f}{}".format(
                        label or "the call", quote.usd, self.spent_usd, self.reserved_usd,
                        self.limit_usd,
                        " (quote uses a conservative rate for unknown pricing)"
                        if quote.conservative else ""),
                    quote=quote)
        reservation = _Reservation(id=uuid.uuid4().hex[:10], usd=quote.usd, label=label,
                                   conservative=quote.conservative)
        self._open[reservation.id] = reservation
        return reservation.id

    def settle(self, reservation_id: str, actual_usd: float) -> float:
        """Convert a reservation into real spend and release the rest."""
        reservation = self._open.pop(reservation_id, None)
        if reservation is None:
            raise KeyError("unknown reservation {!r}".format(reservation_id))
        actual = round(max(0.0, float(actual_usd)), 6)
        self.spent_usd = round(self.spent_usd + actual, 6)
        self.estimated_usd = round(self.estimated_usd + reservation.usd, 6)
        return actual

    def release(self, reservation_id: str) -> None:
        """Give back a reservation for a call that never happened."""
        self._open.pop(reservation_id, None)

    def snapshot(self) -> Dict[str, object]:
        return {"limit_usd": self.limit_usd, "spent_usd": self.spent_usd,
                "reserved_usd": self.reserved_usd, "remaining_usd": self.remaining_usd,
                "estimated_usd": self.estimated_usd, "open_reservations": len(self._open)}

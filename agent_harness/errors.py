"""One error hierarchy, so a caller can tell a config mistake from a model lying.

Provider failures are split by cause because the correct response differs: an
auth failure must not be retried, a rate limit must be, and a malformed body is a
contract problem rather than a transport one.
"""

from __future__ import annotations


class HarnessError(Exception):
    """Base class for everything this package raises on purpose."""


class ConfigError(HarnessError):
    """A profile, roster or example file is malformed or inconsistent."""


class MissingDependency(HarnessError):
    """An optional extra is needed for the path being taken."""


# --- provider failures, by cause ---------------------------------------------
class ProviderError(HarnessError):
    """Any failure originating in a provider."""


class ProviderAuthError(ProviderError):
    """Credentials rejected. Retrying cannot help."""


class ProviderRateLimit(ProviderError):
    """Throttled. Retryable, and it costs budget that must be accounted for."""


class ProviderUnavailable(ProviderError):
    """Transport failure, timeout or 5xx. Retryable within bounds."""


class ProviderTimeout(ProviderUnavailable):
    """The request exceeded its timeout."""


class ProviderBadResponse(ProviderError):
    """The provider answered with something outside the agreed shape."""


class ProviderModelNotFound(ProviderError):
    """The requested model is not provisioned or has reached end of life."""


class ProviderInvalidRequest(ProviderError):
    """The provider rejected the request itself. Retrying cannot help."""


# --- control-flow and state failures -----------------------------------------
class ContractViolation(HarnessError):
    """A model returned something outside the agreed JSON contract."""


class BudgetExceeded(HarnessError):
    """A pre-call budget reservation was refused. Carries the partial run."""

    def __init__(self, message: str, trace=None, quote=None):
        super().__init__(message)
        self.trace = trace
        self.quote = quote


class ReservationError(HarnessError):
    """A reservation was settled or released twice, or never existed."""


class PermissionDenied(HarnessError):
    """An actor attempted a blackboard write it is not authorised to make."""

    def __init__(self, message: str, agent: str = "", key: str = "", reason: str = ""):
        super().__init__(message)
        self.agent = agent
        self.key = key
        self.reason = reason


class OverwriteRejected(HarnessError):
    """An actor tried to replace another actor's artifact without authority."""

    def __init__(self, message: str, key: str = "", owner: str = "", actor: str = ""):
        super().__init__(message)
        self.key = key
        self.owner = owner
        self.actor = actor


class DependencyError(HarnessError):
    """A step was about to run without a required input present."""

    def __init__(self, message: str, step_id: str = "", missing=()):
        super().__init__(message)
        self.step_id = step_id
        self.missing = tuple(missing)

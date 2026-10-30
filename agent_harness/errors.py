"""One error hierarchy, so a caller can tell a config mistake from a model lying."""

from __future__ import annotations


class HarnessError(Exception):
    """Base class for everything this package raises on purpose."""


class ConfigError(HarnessError):
    """A profile, roster or example file is malformed or inconsistent."""


class MissingDependency(HarnessError):
    """An optional extra is needed for the path being taken."""


class ProviderError(HarnessError):
    """The provider failed: transport, auth or a malformed response."""


class ContractViolation(HarnessError):
    """A model returned something outside the agreed JSON contract."""


class BudgetExceeded(HarnessError):
    """A step, cost or time ceiling was hit. Carries the partial run."""

    def __init__(self, message: str, trace=None):
        super().__init__(message)
        self.trace = trace

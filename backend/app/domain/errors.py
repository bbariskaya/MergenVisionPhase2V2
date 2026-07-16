"""Domain-level exceptions."""


class DomainError(Exception):
    """Base class for all domain errors."""


class ValidationError(DomainError):
    """Input or state validation failed."""


class InvalidTransitionError(DomainError):
    """Requested state transition is not allowed."""


class IdentityResolutionError(DomainError):
    """Identity resolution failed after persistence of failure state."""


class ConcurrentUpdateError(DomainError):
    """Optimistic version check failed."""

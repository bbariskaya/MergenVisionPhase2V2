"""Domain-level exceptions."""


class DomainError(Exception):
    """Base class for all domain errors."""


class ValidationError(DomainError):
    """Input or state validation failed."""


class PayloadTooLargeError(DomainError):
    """Input payload exceeds the configured byte limit."""


class UnsupportedMediaTypeError(DomainError):
    """Input media type is not supported."""


class InvalidMediaError(DomainError):
    """Input media failed validation (corrupt, dimensions, pixel limit, etc.)."""


class InvalidTransitionError(DomainError):
    """Requested state transition is not allowed."""


class IdentityResolutionError(DomainError):
    """Identity resolution failed after persistence of failure state."""


class ConcurrentUpdateError(DomainError):
    """Optimistic version check failed."""


class IdempotencyConflictError(DomainError):
    """Idempotency key reused with a different request."""


class JobNotFoundError(DomainError):
    """Requested video job does not exist."""


class VideoNotFoundError(DomainError):
    """Requested video asset does not exist."""


class VideoExecutorNotReadyError(DomainError):
    """Native video executor / common device pipeline is not available."""

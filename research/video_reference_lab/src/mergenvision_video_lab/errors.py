"""Structured error types for the video reference lab."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    """Exit/error codes returned by CLI commands."""

    OK = "OK"
    BLOCKED_PREVIOUS_SPRINT_NOT_CLOSED = "BLOCKED_PREVIOUS_SPRINT_NOT_CLOSED"
    BLOCKED_REFERENCE_ENVIRONMENT = "BLOCKED_REFERENCE_ENVIRONMENT"
    BLOCKED_MODEL_ARTIFACT = "BLOCKED_MODEL_ARTIFACT"
    BLOCKED_INPUT_VIDEO = "BLOCKED_INPUT_VIDEO"
    BLOCKED_RECOGNIZER_DETERMINISM = "BLOCKED_RECOGNIZER_DETERMINISM"
    PARTIAL_NEEDS_HUMAN_LABELS = "PARTIAL_NEEDS_HUMAN_LABELS"
    CONFIG_INVALID = "CONFIG_INVALID"
    VIDEO_UNREADABLE = "VIDEO_UNREADABLE"
    ARTIFACT_CORRUPT = "ARTIFACT_CORRUPT"
    TRACKER_ORDER_VIOLATION = "TRACKER_ORDER_VIOLATION"
    CANNOT_LINK_VIOLATION = "CANNOT_LINK_VIOLATION"


class LabError(Exception):
    """Base class for lab errors with a structured code."""

    def __init__(
        self, code: ErrorCode, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


class ConfigError(LabError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.CONFIG_INVALID, message, details)


class ModelArtifactError(LabError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.BLOCKED_MODEL_ARTIFACT, message, details)


class VideoReadError(LabError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.VIDEO_UNREADABLE, message, details)


class ArtifactCorruptError(LabError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.ARTIFACT_CORRUPT, message, details)

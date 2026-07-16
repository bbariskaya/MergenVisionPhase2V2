"""Tracker protocol and shared track state definitions."""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import TYPE_CHECKING, Protocol

import numpy as np

from mergenvision_video_lab.contracts import FaceObservation, TrackAssignment

if TYPE_CHECKING:
    from mergenvision_video_lab.tracking.byte_tracker import Tracklet


class TrackState(IntEnum):
    """Lifecycle state of a raw tracklet."""

    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class MetadataTracker(Protocol):
    """Protocol for a deterministic sequential face metadata tracker."""

    @property
    def strategy(self) -> str: ...

    def update(
        self,
        frame_index: int,
        pts_ns: int,
        observations: Sequence[FaceObservation],
        embeddings: np.ndarray,
        scene_cut_before: bool,
    ) -> Sequence[TrackAssignment]:
        """Consume one frame's observations and emit assignments."""
        ...

    def finalize(self) -> None:
        """Mark all active/lost tracks as removed."""
        ...

    def active_tracklet_ids(self) -> list[str]: ...

    def removed_tracklets(self) -> list[Tracklet]: ...

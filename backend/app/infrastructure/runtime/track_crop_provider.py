"""Placeholder crop provider for video tracks.

This adapter satisfies the TrackCropProvider port while the real per-track
best-shot extraction from the decoded video source is being implemented.
Production code should replace it with a proper frame/crop extractor.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.domain.entities.video_tracking import CanonicalTrack


class PlaceholderTrackCropProvider:
    def __init__(self, payload: bytes = b"placeholder-crop") -> None:
        self._payload = payload

    async def get_crop(
        self,
        _track_id: uuid.UUID,
        track: CanonicalTrack | None = None,
    ) -> bytes:
        return self._payload

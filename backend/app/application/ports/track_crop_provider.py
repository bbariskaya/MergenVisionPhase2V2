"""Port for providing a crop image for a canonical video track."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from app.domain.entities.video_tracking import CanonicalTrack


class TrackCropProvider(Protocol):
    async def get_crop(
        self,
        track_id: uuid.UUID,
        track: CanonicalTrack | None = None,
    ) -> bytes: ...

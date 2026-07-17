from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from app.infrastructure.serialization.native_bundle_reader import NativeBundle, NativeBundleError

if TYPE_CHECKING:
    from app.domain.entities.video_tracking import CanonicalTrack


class NativeRepresentativeCropProvider:
    """Artifact-based crop provider.

    Reads the native worker output bundle and returns the representative aligned
    WebP crop for a canonical track by selecting the best raw-track candidate in
    the canonical cluster.
    """

    def __init__(self, bundle_dir: Path) -> None:
        self._bundle = NativeBundle(bundle_dir)

    async def get_crop(self, track_id: uuid.UUID, track: CanonicalTrack | None = None) -> bytes:
        if track is None:
            raise NativeBundleError(
                f"NativeRepresentativeCropProvider requires track context for {track_id}"
            )
        raw_keys = self._collect_raw_keys(track)
        if not raw_keys:
            raise NativeBundleError(f"no raw-track keys for canonical track {track_id}")

        best_key = self._select_best_raw_track(raw_keys)
        return self._bundle.read_crop_bytes(best_key)

    def _collect_raw_keys(self, track: CanonicalTrack) -> set[str]:
        keys: set[str] = set()
        for tracklet in track.tracklets:
            for det in tracklet.detections:
                if det.raw_track_key:
                    keys.add(det.raw_track_key)
        return keys

    def _select_best_raw_track(self, raw_keys: set[str]) -> str:
        candidates = []
        for key in raw_keys:
            template = self._bundle.template_for(key)
            if template is None:
                continue
            has_crop = bool(template.representative_crop_relative_key)
            quality = template.template_quality if template.template_quality > 0 else 0.0
            candidates.append((key, has_crop, quality, template.observation_count))

        if not candidates:
            raise NativeBundleError("no raw-track templates found for canonical track")

        # Prefer templates with a crop, then highest quality, then most observations,
        # then deterministic key ordering.
        candidates.sort(key=lambda c: (-int(c[1]), -c[2], -c[3], c[0]))
        return candidates[0][0]

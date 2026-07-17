from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.infrastructure.serialization.video_track_template_reader import (
    read_track_templates,
)


class NativeBundleError(Exception):
    pass


def _webp_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise NativeBundleError("not a valid WebP file")
    chunk = data[12:16]
    if chunk == b"VP8 ":
        # Simple lossy WebP: VP8 data starts after the 8-byte chunk header.
        # The key frame layout is: 3-byte frame tag, 3-byte start code (0x9d 0x01 0x2a),
        # then LE 16-bit width/height at offsets 26 and 28 (absolute).
        if len(data) < 30:
            raise NativeBundleError("VP8 data too short")
        width = int.from_bytes(data[26:28], "little") & 0x3FFF
        height = int.from_bytes(data[28:30], "little") & 0x3FFF
        return width, height
    if chunk == b"VP8L":
        # Lossless WebP: dimensions packed in 24 bits at offset 21..23
        b0, b1, b2, b3 = data[21:25]
        width = 1 + (((b1 & 0x3F) << 8) | b0)
        height = 1 + (((b3 & 0xF) << 10) | (b2 << 2) | ((b1 & 0xC0) >> 6))
        return width, height
    if chunk == b"VP8X":
        # Extended WebP: canvas size is 24-bit big-endian at offsets 24..26 (w) and 27..29 (h).
        width = 1 + int.from_bytes(data[24:27], "big")
        height = 1 + int.from_bytes(data[27:30], "big")
        return width, height
    raise NativeBundleError("unsupported WebP chunk type")


class NativeBundle:
    def __init__(self, bundle_dir: Path) -> None:
        self._dir = bundle_dir
        manifest_path = bundle_dir / "manifest.json"
        if not manifest_path.exists():
            raise NativeBundleError(f"manifest.json missing in {bundle_dir}")
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))

        obs_path = bundle_dir / "observations.pb"
        if not obs_path.exists():
            zst_path = bundle_dir / "observations.pb.zst"
            if zst_path.exists():
                obs_path = zst_path
            else:
                raise NativeBundleError("observations.pb(.zst) missing")
        self.observation_path = obs_path

        tmpl_path = bundle_dir / "track_templates.pb"
        if not tmpl_path.exists():
            zst_path = bundle_dir / "track_templates.pb.zst"
            if zst_path.exists():
                tmpl_path = zst_path
            else:
                raise NativeBundleError("track_templates.pb(.zst) missing")
        self.template_path = tmpl_path

        self._templates = {t.raw_track_key: t for t in read_track_templates(tmpl_path)}
        self._crop_dir = bundle_dir / "crops"
        self._validate_counts()

    @property
    def raw_track_keys(self) -> set[str]:
        return set(self._templates.keys())

    def template_for(self, raw_track_key: str) -> Any:
        return self._templates.get(raw_track_key)

    def read_crop_bytes(self, raw_track_key: str) -> bytes:
        template = self._templates.get(raw_track_key)
        if template is None:
            raise NativeBundleError(f"unknown raw track key: {raw_track_key}")
        rel = template.representative_crop_relative_key
        if not rel:
            raise NativeBundleError(f"no representative crop for {raw_track_key}")

        crop_path = self._crop_dir / Path(rel).name
        if not crop_path.exists():
            raise NativeBundleError(f"crop file missing: {crop_path}")

        data = crop_path.read_bytes()
        width, height = _webp_dimensions(data)
        if width != 112 or height != 112:
            raise NativeBundleError(
                f"crop {crop_path} has invalid dimensions {width}x{height}"
            )
        return data

    def sha256(self, relative_path: str) -> str:
        path = self._dir / relative_path
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _validate_counts(self) -> None:
        manifest_crop_count = self.manifest.get("crop_count", 0)
        if manifest_crop_count < 0:
            raise NativeBundleError("invalid crop_count in manifest")
        actual_crop_count = sum(
            1 for t in self._templates.values() if t.representative_crop_relative_key
        )
        if actual_crop_count != manifest_crop_count:
            raise NativeBundleError(
                f"manifest crop_count {manifest_crop_count} != actual {actual_crop_count}"
            )

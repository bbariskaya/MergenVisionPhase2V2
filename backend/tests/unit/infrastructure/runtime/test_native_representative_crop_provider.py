import io
import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.domain.entities.video_tracking import CanonicalTrack, RawTracklet, TrackDetection
from app.domain.value_objects import BoundingBox
from app.infrastructure.runtime.native_representative_crop_provider import (
    NativeBundleError,
    NativeRepresentativeCropProvider,
)
from app.infrastructure.serialization import (
    ObservationChunkFooter,
    RawTrackTemplate,
    TrackTemplateBundle,
    TrackTemplateFooter,
    VideoObservationFrame,
)


def _varint_bytes(value: int) -> bytes:
    from google.protobuf.internal.encoder import _EncodeVarint

    out = io.BytesIO()
    _EncodeVarint(out.write, value)
    return out.getvalue()


def _webp112() -> bytes:
    data = bytearray(32)
    data[:4] = b"RIFF"
    data[8:12] = b"WEBP"
    data[12:16] = b"VP8 "
    file_size = len(data) - 8
    data[4:8] = file_size.to_bytes(4, "little")
    chunk_size = len(data) - 20
    data[16:20] = chunk_size.to_bytes(4, "little")
    data[20:23] = b"\xd0\x01\x00"
    data[23:26] = b"\x9d\x01\x2a"
    data[26] = 112 & 0xFF
    data[27] = (112 >> 8) & 0x3F
    data[28] = 112 & 0xFF
    data[29] = (112 >> 8) & 0x3F
    return bytes(data)


def _write_bundle(bundle_dir: Path, raw_key: str, quality: float = 0.8) -> None:
    (bundle_dir / "crops").mkdir(exist_ok=True)
    job_id = str(uuid4())

    obs_path = bundle_dir / "observations.pb"
    with obs_path.open("wb") as f:
        frame = VideoObservationFrame()
        frame.job_id = job_id
        frame.video_id = str(uuid4())
        frame.frame_index = 0
        payload = frame.SerializeToString()
        f.write(_varint_bytes(len(payload)) + payload)
        footer = ObservationChunkFooter()
        footer.job_id = job_id
        footer.frame_count = 1
        payload = footer.SerializeToString()
        f.write(_varint_bytes(len(payload)) + payload)

    tmpl_path = bundle_dir / "track_templates.pb"
    bundle = TrackTemplateBundle()
    bundle.job_id = job_id
    bundle.video_id = str(uuid4())
    t = RawTrackTemplate()
    t.raw_track_key = raw_key
    t.first_pts_ns = 0
    t.last_pts_ns = 33_000_000
    t.observation_count = 1
    t.eligible_observation_count = 1
    t.template_embedding.extend([0.0] * 512)
    t.template_quality = quality
    t.representative_crop_relative_key = f"crops/{raw_key}.webp"
    t.representative_pts_ns = 16_000_000
    t.representative_ordinal = 0
    footer = TrackTemplateFooter()
    footer.job_id = job_id
    footer.template_count = 1
    with tmpl_path.open("wb") as f:
        payload = bundle.SerializeToString()
        f.write(_varint_bytes(len(payload)) + payload)
        payload = t.SerializeToString()
        f.write(_varint_bytes(len(payload)) + payload)
        payload = footer.SerializeToString()
        f.write(_varint_bytes(len(payload)) + payload)

    (bundle_dir / "crops" / f"{raw_key}.webp").write_bytes(_webp112())

    manifest = {
        "schema_versions": {"manifest": "1"},
        "job_id": job_id,
        "video_id": str(uuid4()),
        "crop_count": 1,
        "raw_track_count": 1,
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _canonical_track(raw_key: str) -> CanonicalTrack:
    track = CanonicalTrack(track_id=uuid4())
    tracklet = RawTracklet(
        tracklet_id=uuid4(),
        job_id=uuid4(),
        ordinal=0,
    )
    tracklet.detections.append(
        TrackDetection(
            detection_id="det-0",
            frame_index=0,
            pts_ns=0,
            bbox=BoundingBox(x=10, y=10, width=80, height=90),
            landmarks=tuple(10.0 for _ in range(10)),
            detector_score=0.95,
            quality_score=0.9,
            embedding=tuple(0.0 for _ in range(512)),
            raw_track_key=raw_key,
        )
    )
    track.tracklets.append(tracklet)
    return track


@pytest.mark.asyncio
async def test_get_crop_returns_representative_webp(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _write_bundle(bundle_dir, "raw-best")

    provider = NativeRepresentativeCropProvider(bundle_dir)
    track = _canonical_track("raw-best")
    crop = await provider.get_crop(track.track_id, track=track)
    assert crop[:4] == b"RIFF"


@pytest.mark.asyncio
async def test_get_crop_selects_best_quality_candidate(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _write_bundle(bundle_dir, "raw-high", quality=0.95)
    _write_bundle(bundle_dir, "raw-low", quality=0.10)
    # Second call overwrites files; we need both raw tracks in one bundle.
    # Rewrite a proper bundle with two raw tracks.
    (bundle_dir / "track_templates.pb").unlink(missing_ok=True)
    (bundle_dir / "manifest.json").unlink(missing_ok=True)
    (bundle_dir / "crops").mkdir(exist_ok=True)
    job_id = str(uuid4())

    t_high = RawTrackTemplate()
    t_high.raw_track_key = "raw-high"
    t_high.observation_count = 1
    t_high.eligible_observation_count = 1
    t_high.template_embedding.extend([0.0] * 512)
    t_high.template_quality = 0.95
    t_high.representative_crop_relative_key = "crops/raw-high.webp"

    t_low = RawTrackTemplate()
    t_low.raw_track_key = "raw-low"
    t_low.observation_count = 1
    t_low.eligible_observation_count = 1
    t_low.template_embedding.extend([0.0] * 512)
    t_low.template_quality = 0.10
    t_low.representative_crop_relative_key = "crops/raw-low.webp"

    bundle = TrackTemplateBundle()
    bundle.job_id = job_id
    bundle.video_id = str(uuid4())
    footer = TrackTemplateFooter()
    footer.job_id = job_id
    footer.template_count = 2
    with (bundle_dir / "track_templates.pb").open("wb") as f:
        payload = bundle.SerializeToString()
        f.write(_varint_bytes(len(payload)) + payload)
        payload = t_high.SerializeToString()
        f.write(_varint_bytes(len(payload)) + payload)
        payload = t_low.SerializeToString()
        f.write(_varint_bytes(len(payload)) + payload)
        payload = footer.SerializeToString()
        f.write(_varint_bytes(len(payload)) + payload)

    (bundle_dir / "crops" / "raw-high.webp").write_bytes(_webp112())
    (bundle_dir / "crops" / "raw-low.webp").write_bytes(_webp112())
    manifest = {
        "schema_versions": {"manifest": "1"},
        "job_id": job_id,
        "video_id": str(uuid4()),
        "crop_count": 2,
        "raw_track_count": 2,
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    track = CanonicalTrack(track_id=uuid4())
    for raw_key in ("raw-high", "raw-low"):
        tracklet = RawTracklet(tracklet_id=uuid4(), job_id=uuid4(), ordinal=0)
        tracklet.detections.append(
            TrackDetection(
                detection_id=f"det-{raw_key}",
                frame_index=0,
                pts_ns=0,
                bbox=BoundingBox(x=10, y=10, width=80, height=90),
                landmarks=tuple(10.0 for _ in range(10)),
                detector_score=0.95,
                quality_score=0.9,
                embedding=tuple(0.0 for _ in range(512)),
                raw_track_key=raw_key,
            )
        )
        track.tracklets.append(tracklet)

    provider = NativeRepresentativeCropProvider(bundle_dir)
    crop = await provider.get_crop(track.track_id, track=track)
    # Provider selects raw-high because its template_quality is higher.
    selected = provider._select_best_raw_track({"raw-high", "raw-low"})
    assert selected == "raw-high"
    assert crop[:4] == b"RIFF"


@pytest.mark.asyncio
async def test_get_crop_requires_track_context(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _write_bundle(bundle_dir, "raw-only")

    provider = NativeRepresentativeCropProvider(bundle_dir)
    with pytest.raises(NativeBundleError, match="requires track context"):
        await provider.get_crop(uuid4())


@pytest.mark.asyncio
async def test_get_crop_fails_when_no_raw_keys_match(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _write_bundle(bundle_dir, "raw-a")

    provider = NativeRepresentativeCropProvider(bundle_dir)
    track = _canonical_track("raw-b")
    with pytest.raises(NativeBundleError, match="no raw-track templates found"):
        await provider.get_crop(track.track_id, track=track)

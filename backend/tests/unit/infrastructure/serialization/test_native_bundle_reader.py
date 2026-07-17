import io
import json
from pathlib import Path
from uuid import uuid4

import pytest
import zstandard

from app.infrastructure.serialization import (
    ObservationChunkFooter,
    RawTrackTemplate,
    TrackTemplateBundle,
    TrackTemplateFooter,
    VideoObservationFrame,
)
from app.infrastructure.serialization.native_bundle_reader import (
    NativeBundle,
    NativeBundleError,
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
    # Minimal VP8 key-frame tag + start code.
    data[20:23] = b"\xd0\x01\x00"
    data[23:26] = b"\x9d\x01\x2a"
    width = 112
    height = 112
    data[26] = width & 0xFF
    data[27] = (width >> 8) & 0x3F
    data[28] = height & 0xFF
    data[29] = (height >> 8) & 0x3F
    return bytes(data)


def _write_observations(path: Path, job_id: str, frame_count: int = 1) -> None:
    with path.open("wb") as f:
        for i in range(frame_count):
            frame = VideoObservationFrame()
            frame.job_id = job_id
            frame.video_id = str(uuid4())
            frame.frame_index = i
            frame.pts_ns = i * 33_000_000
            payload = frame.SerializeToString()
            f.write(_varint_bytes(len(payload)) + payload)
        footer = ObservationChunkFooter()
        footer.job_id = job_id
        footer.sequence_no = 0
        footer.frame_count = frame_count
        payload = footer.SerializeToString()
        f.write(_varint_bytes(len(payload)) + payload)


def _write_templates(path: Path, job_id: str, templates: list[RawTrackTemplate]) -> None:
    bundle = TrackTemplateBundle()
    bundle.job_id = job_id
    bundle.video_id = str(uuid4())
    bundle.sequence_no = 0
    bundle.model_version = "retinaface_r50_glintr100_v1"
    bundle.preprocess_version = "cuda_five_point_align"
    bundle.config_version = "1"
    with path.open("wb") as f:
        payload = bundle.SerializeToString()
        f.write(_varint_bytes(len(payload)) + payload)
        for t in templates:
            payload = t.SerializeToString()
            f.write(_varint_bytes(len(payload)) + payload)
        footer = TrackTemplateFooter()
        footer.job_id = job_id
        footer.sequence_no = 0
        footer.template_count = len(templates)
        payload = footer.SerializeToString()
        f.write(_varint_bytes(len(payload)) + payload)


def _build_template(raw_track_key: str, with_crop: bool = True) -> RawTrackTemplate:
    t = RawTrackTemplate()
    t.raw_track_key = raw_track_key
    t.first_pts_ns = 1_000_000
    t.last_pts_ns = 2_000_000
    t.observation_count = 3
    t.eligible_observation_count = 2
    t.template_embedding.extend([0.0] * 512)
    t.template_quality = 0.85
    if with_crop:
        t.representative_crop_relative_key = f"crops/{raw_track_key}.webp"
    t.representative_pts_ns = 1_500_000
    t.representative_ordinal = 1
    return t


def _write_manifest(bundle_dir: Path, *, crop_count: int, raw_track_count: int = 1) -> None:
    manifest = {
        "schema_versions": {"manifest": "1"},
        "job_id": str(uuid4()),
        "video_id": str(uuid4()),
        "crop_count": crop_count,
        "raw_track_count": raw_track_count,
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _valid_bundle(tmp_path: Path, compress: bool = False) -> Path:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "crops").mkdir()
    job_id = str(uuid4())
    template = _build_template("raw-0001")
    if compress:
        obs_raw = tmp_path / "obs.pb"
        tmpl_raw = tmp_path / "tmpl.pb"
        _write_observations(obs_raw, job_id)
        _write_templates(tmpl_raw, job_id, [template])
        (bundle_dir / "observations.pb.zst").write_bytes(
            zstandard.ZstdCompressor().compress(obs_raw.read_bytes())
        )
        (bundle_dir / "track_templates.pb.zst").write_bytes(
            zstandard.ZstdCompressor().compress(tmpl_raw.read_bytes())
        )
    else:
        _write_observations(bundle_dir / "observations.pb", job_id)
        _write_templates(bundle_dir / "track_templates.pb", job_id, [template])
    (bundle_dir / "crops" / "raw-0001.webp").write_bytes(_webp112())
    _write_manifest(bundle_dir, crop_count=1)
    return bundle_dir


def test_reads_uncompressed_bundle(tmp_path: Path) -> None:
    bundle_dir = _valid_bundle(tmp_path, compress=False)
    bundle = NativeBundle(bundle_dir)
    assert bundle.raw_track_keys == {"raw-0001"}
    assert bundle.template_for("raw-0001") is not None
    crop = bundle.read_crop_bytes("raw-0001")
    assert crop[:4] == b"RIFF"


def test_reads_zstd_compressed_bundle(tmp_path: Path) -> None:
    bundle_dir = _valid_bundle(tmp_path, compress=True)
    bundle = NativeBundle(bundle_dir)
    assert "observations.pb.zst" in str(bundle.observation_path)
    assert bundle.raw_track_keys == {"raw-0001"}
    crop = bundle.read_crop_bytes("raw-0001")
    assert crop[:4] == b"RIFF"


def test_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(NativeBundleError, match="manifest.json missing"):
        NativeBundle(tmp_path / "missing")


def test_missing_observations_raises(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _write_manifest(bundle_dir, crop_count=0)
    with pytest.raises(NativeBundleError, match="observations.pb"):
        NativeBundle(bundle_dir)


def test_missing_crop_raises(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "crops").mkdir()
    job_id = str(uuid4())
    _write_observations(bundle_dir / "observations.pb", job_id)
    template = _build_template("raw-0001")
    _write_templates(bundle_dir / "track_templates.pb", job_id, [template])
    _write_manifest(bundle_dir, crop_count=1)
    bundle = NativeBundle(bundle_dir)
    with pytest.raises(NativeBundleError, match="crop file missing"):
        bundle.read_crop_bytes("raw-0001")


def test_invalid_crop_dimensions_raise(tmp_path: Path) -> None:
    bundle_dir = _valid_bundle(tmp_path, compress=False)
    crop_path = bundle_dir / "crops" / "raw-0001.webp"
    bad = bytearray(_webp112())
    bad[26] = 64  # change width decode to 64
    bad[28] = 64
    crop_path.write_bytes(bytes(bad))
    bundle = NativeBundle(bundle_dir)
    with pytest.raises(NativeBundleError, match="invalid dimensions"):
        bundle.read_crop_bytes("raw-0001")


def test_crop_count_mismatch_raises(tmp_path: Path) -> None:
    bundle_dir = _valid_bundle(tmp_path, compress=False)
    _write_manifest(bundle_dir, crop_count=99)
    with pytest.raises(NativeBundleError, match="crop_count"):
        NativeBundle(bundle_dir)


def test_sha256_file(tmp_path: Path) -> None:
    bundle_dir = _valid_bundle(tmp_path, compress=False)
    (bundle_dir / "extra.bin").write_bytes(b"hello")
    bundle = NativeBundle(bundle_dir)
    assert bundle.sha256("extra.bin") == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

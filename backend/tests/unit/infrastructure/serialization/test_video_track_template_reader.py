import io
from pathlib import Path
from uuid import uuid4

import pytest
import zstandard

from app.infrastructure.serialization import (
    RawTrackTemplate,
    TrackTemplateBundle,
    TrackTemplateFooter,
)
from app.infrastructure.serialization.video_track_template_reader import (
    TrackTemplateArtifactError,
    read_track_templates,
)


def _varint_bytes(value: int) -> bytes:
    from google.protobuf.internal.encoder import _EncodeVarint

    out = io.BytesIO()
    _EncodeVarint(out.write, value)
    return out.getvalue()


def _build_bundle(job_id: str, video_id: str) -> TrackTemplateBundle:
    bundle = TrackTemplateBundle()
    bundle.job_id = job_id
    bundle.video_id = video_id
    bundle.sequence_no = 0
    bundle.model_version = "retinaface_r50_glintr100_v1"
    bundle.preprocess_version = "cuda_five_point_align"
    bundle.config_version = "1"
    return bundle


def _build_template(raw_track_key: str) -> RawTrackTemplate:
    t = RawTrackTemplate()
    t.raw_track_key = raw_track_key
    t.first_pts_ns = 1_000_000
    t.last_pts_ns = 2_000_000
    t.observation_count = 5
    t.eligible_observation_count = 3
    t.template_embedding.extend([0.0] * 512)
    t.template_quality = 0.91
    t.representative_crop_relative_key = f"crops/{raw_track_key}.webp"
    t.representative_pts_ns = 1_500_000
    t.representative_ordinal = 2
    return t


def _write_bundle(path: Path, bundle: TrackTemplateBundle, templates: list[RawTrackTemplate]) -> None:
    with path.open("wb") as f:
        payload = bundle.SerializeToString()
        f.write(_varint_bytes(len(payload)) + payload)
        for t in templates:
            payload = t.SerializeToString()
            f.write(_varint_bytes(len(payload)) + payload)
        footer = TrackTemplateFooter()
        footer.job_id = bundle.job_id
        footer.sequence_no = 0
        footer.template_count = len(templates)
        payload = footer.SerializeToString()
        f.write(_varint_bytes(len(payload)) + payload)


def test_read_uncompressed_templates(tmp_path: Path) -> None:
    job_id = str(uuid4())
    video_id = str(uuid4())
    path = tmp_path / "templates.pb"
    bundle = _build_bundle(job_id, video_id)
    template = _build_template("raw-0001")
    _write_bundle(path, bundle, [template])

    templates = read_track_templates(path)
    assert len(templates) == 1
    assert templates[0].raw_track_key == "raw-0001"
    assert len(templates[0].template_embedding) == 512
    assert templates[0].template_quality == pytest.approx(0.91)


def test_read_zstd_compressed_templates(tmp_path: Path) -> None:
    job_id = str(uuid4())
    video_id = str(uuid4())
    path = tmp_path / "templates.pb.zst"
    raw_path = tmp_path / "templates.pb"
    _write_bundle(raw_path, _build_bundle(job_id, video_id), [_build_template("raw-0002")])
    compressed = zstandard.ZstdCompressor().compress(raw_path.read_bytes())
    path.write_bytes(compressed)

    templates = read_track_templates(path)
    assert len(templates) == 1
    assert templates[0].raw_track_key == "raw-0002"


def test_missing_bundle_header_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.pb"
    with path.open("wb") as f:
        # A deliberately malformed first message that cannot be parsed as a bundle.
        f.write(_varint_bytes(4) + b"\xff\xff\xff\xff")
    with pytest.raises(TrackTemplateArtifactError, match="missing TrackTemplateBundle header"):
        read_track_templates(path)


def test_footer_count_mismatch_raises(tmp_path: Path) -> None:
    job_id = str(uuid4())
    path = tmp_path / "bad.pb"
    bundle = _build_bundle(job_id, str(uuid4()))
    with path.open("wb") as f:
        payload = bundle.SerializeToString()
        f.write(_varint_bytes(len(payload)) + payload)
        footer = TrackTemplateFooter()
        footer.job_id = job_id
        footer.template_count = 99
        payload = footer.SerializeToString()
        f.write(_varint_bytes(len(payload)) + payload)
    with pytest.raises(TrackTemplateArtifactError, match="footer template_count"):
        read_track_templates(path)

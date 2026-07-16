"""Unit tests for artifact store."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mergenvision_video_lab.artifact_store import ArtifactStore
from mergenvision_video_lab.contracts import (
    AlignmentContract,
    BBoxXYXY,
    FaceObservation,
    Landmarks5,
    OnnxModelContract,
    QualityMetrics,
    RunManifest,
    SamplingContract,
)
from mergenvision_video_lab.errors import ArtifactCorruptError


def _dummy_manifest(obs_count: int = 1, emb_count: int = 1) -> RunManifest:
    return RunManifest(
        run_id="run_1",
        video_sha256="a" * 64,
        video_size_bytes=1000,
        logical_video_name="test.mp4",
        container="mp4",
        codec="h264",
        pixel_format="yuv420p",
        display_width=100,
        display_height=100,
        rotation_degrees=0.0,
        stream_index=0,
        time_base_num=1,
        time_base_den=30,
        duration_ns=1_000_000_000,
        decoded_frame_count=30,
        processed_frame_count=30,
        sampling_contract=SamplingContract(mode="every_frame"),
        detector_contract=OnnxModelContract(
            basename="det.onnx", sha256="b" * 64, size_bytes=1, inputs=[], outputs=[]
        ),
        recognizer_contract=OnnxModelContract(
            basename="rec.onnx", sha256="c" * 64, size_bytes=1, inputs=[], outputs=[]
        ),
        alignment_contract=AlignmentContract(
            output_size=112,
            color_order="BGR",
            border_mode="constant_zero",
            interpolation="bilinear",
            landmark_order=[],
            arcface_template=[],
        ),
        provider_requested="cpu",
        providers_available=["CPUExecutionProvider"],
        providers_actual=["CPUExecutionProvider"],
        package_versions={},
        config_sha256="d" * 64,
        observation_count=obs_count,
        valid_embedding_count=emb_count,
        rejection_counts={},
        extraction_timing={
            "decode_seconds": 0.1,
            "oracle_seconds": 0.2,
            "quality_alignment_seconds": 0.1,
            "serialization_seconds": 0.1,
            "total_seconds": 0.5,
        },
    )


def _dummy_observation(embedding_index: int | None = 0) -> FaceObservation:
    return FaceObservation(
        observation_id="obs_1",
        source_id="test.mp4",
        frame_index=0,
        pts=0,
        time_base_num=1,
        time_base_den=30,
        pts_ns=0,
        frame_width=100,
        frame_height=100,
        detection_ordinal=0,
        bbox_xyxy=BBoxXYXY(x1=10.0, y1=10.0, x2=50.0, y2=50.0),
        detector_score=0.9,
        landmarks_5=Landmarks5(
            left_eye=(20.0, 20.0),
            right_eye=(40.0, 20.0),
            nose=(30.0, 30.0),
            left_mouth=(22.0, 40.0),
            right_mouth=(38.0, 40.0),
        ),
        quality=QualityMetrics(
            bbox_width_px=40.0,
            bbox_height_px=40.0,
            bbox_min_side_px=40.0,
            bbox_area_px=1600.0,
            detector_score=0.9,
            grayscale_laplacian_variance=100.0,
            brightness_mean=128.0,
            brightness_std=30.0,
            dark_clip_fraction=0.0,
            bright_clip_fraction=0.0,
            interocular_distance_px=20.0,
            alignment_reprojection_error_px=1.0,
            alignment_error_normalized_by_interocular=0.05,
            landmark_geometry_valid=True,
            finite_embedding=True,
            composite_quality_score=0.8,
        ),
        tracking_eligible=True,
        recognition_eligible=True,
        embedding_index=embedding_index,
    )


def test_write_and_read_manifest(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "run")
    manifest = _dummy_manifest()
    store.write_manifest(manifest)
    loaded = store.read_manifest()
    assert loaded.run_id == "run_1"


def test_write_and_read_observations(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "run")
    obs = [_dummy_observation()]
    store.write_observations(obs)
    loaded = store.read_observations()
    assert len(loaded) == 1
    assert loaded[0].observation_id == "obs_1"


def test_write_and_read_embeddings(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "run")
    embeddings = np.array([[1.0, 0.0] + [0.0] * 510], dtype=np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    store.write_embeddings(embeddings)
    loaded = store.read_embeddings()
    np.testing.assert_allclose(loaded, embeddings)


def test_validate_complete_run(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "run")
    manifest = _dummy_manifest(obs_count=2, emb_count=1)
    store.write_manifest(manifest)
    store.write_observations([_dummy_observation(0), _dummy_observation(None)])
    emb = np.array([[1.0, 0.0] + [0.0] * 510], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    store.write_embeddings(emb)
    store.write_checksums()
    store.validate()
    assert store.is_complete()


def test_validate_rejects_embedding_index_out_of_range(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "run")
    manifest = _dummy_manifest(obs_count=1, emb_count=1)
    obs = _dummy_observation(embedding_index=5)
    store.write_manifest(manifest)
    store.write_observations([obs])
    emb = np.array([[1.0, 0.0] + [0.0] * 510], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    store.write_embeddings(emb)
    store.write_checksums()
    with pytest.raises(ArtifactCorruptError):
        store.validate()


def test_validate_rejects_non_unit_embeddings(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "run")
    manifest = _dummy_manifest(obs_count=0, emb_count=1)
    store.write_manifest(manifest)
    store.write_observations([])
    store.write_embeddings(np.array([[2.0, 0.0] + [0.0] * 510], dtype=np.float32))
    store.write_checksums()
    with pytest.raises(ArtifactCorruptError):
        store.validate()

"""Unit tests for the annotator backend."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

# Ensure src/ is on path for direct module imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mergenvision_video_lab.annotator_server import create_app
from mergenvision_video_lab.artifact_store import ArtifactStore
from mergenvision_video_lab.config import LabConfig, resolve_run_dir
from mergenvision_video_lab.contracts import (
    AlignmentContract,
    BBoxXYXY,
    ExtractionTiming,
    FaceObservation,
    FrameRecord,
    Landmarks5,
    OnnxModelContract,
    QualityMetrics,
    RunManifest,
    SamplingContract,
)
from mergenvision_video_lab.ground_truth import build_ground_truth_template
from mergenvision_video_lab.hashing import sha256_file


def _face_observation(observation_id: str, frame_index: int) -> FaceObservation:
    return FaceObservation(
        observation_id=observation_id,
        source_id="synthetic",
        frame_index=frame_index,
        pts=frame_index,
        time_base_num=1,
        time_base_den=1,
        pts_ns=frame_index * 1_000_000,
        frame_width=640,
        frame_height=480,
        detection_ordinal=0,
        bbox_xyxy=BBoxXYXY(x1=100.0, y1=100.0, x2=150.0, y2=150.0),
        detector_score=0.8,
        landmarks_5=Landmarks5(
            left_eye=(40.0, 50.0),
            right_eye=(70.0, 50.0),
            nose=(55.0, 70.0),
            left_mouth=(45.0, 90.0),
            right_mouth=(65.0, 90.0),
        ),
        quality=QualityMetrics(
            bbox_width_px=50.0,
            bbox_height_px=50.0,
            bbox_min_side_px=50.0,
            bbox_area_px=2500.0,
            detector_score=0.8,
            grayscale_laplacian_variance=100.0,
            brightness_mean=128.0,
            brightness_std=30.0,
            dark_clip_fraction=0.0,
            bright_clip_fraction=0.0,
            interocular_distance_px=30.0,
            alignment_reprojection_error_px=1.0,
            alignment_error_normalized_by_interocular=0.03,
            landmark_geometry_valid=True,
            finite_embedding=True,
            composite_quality_score=0.9,
        ),
        tracking_eligible=True,
        recognition_eligible=True,
        embedding_index=0,
    )


def _manifest(video_path: Path, video_sha256: str) -> RunManifest:
    align = AlignmentContract(
        output_size=112,
        color_order="BGR",
        border_mode="constant_zero",
        interpolation="bilinear",
        landmark_order=[
            "left_eye",
            "right_eye",
            "nose",
            "left_mouth",
            "right_mouth",
        ],
        arcface_template=[[0.0, 0.0]],
    )
    model = OnnxModelContract(
        basename="dummy.onnx",
        sha256="0" * 64,
        size_bytes=1,
        inputs=[],
        outputs=[],
    )
    return RunManifest(
        run_id=f"{video_sha256[:12]}_test",
        video_sha256=video_sha256,
        video_size_bytes=video_path.stat().st_size,
        logical_video_name="test",
        container="mp4",
        codec="h264",
        pixel_format="yuv420p",
        display_width=640,
        display_height=480,
        rotation_degrees=0.0,
        stream_index=0,
        time_base_num=1,
        time_base_den=1_000_000,
        duration_ns=3_000_000,
        decoded_frame_count=3,
        sampled_frame_count=3,
        processed_frame_count=3,
        sampling_contract=SamplingContract(mode="every_frame"),
        detector_contract=model,
        recognizer_contract=model,
        alignment_contract=align,
        provider_requested="cpu",
        providers_available=["CPUExecutionProvider"],
        providers_actual=["CPUExecutionProvider"],
        package_versions={},
        config_sha256="a" * 64,
        observation_count=1,
        valid_embedding_count=1,
        rejection_counts={},
        extraction_timing=ExtractionTiming(
            decode_seconds=0.1,
            oracle_seconds=0.1,
            quality_alignment_seconds=0.1,
            serialization_seconds=0.1,
            total_seconds=0.4,
        ),
    )


def _write_run_dir(tmp_path: Path) -> tuple[LabConfig, Path]:
    video_path = tmp_path / "test_video.mp4"
    video_path.write_bytes(b"dummy video content")
    video_sha256 = sha256_file(video_path)

    cfg = LabConfig(
        video={
            "path": str(video_path),
            "sampling_mode": "every_frame",
        },
        oracle={"provider": "cpu"},
        output={"base_dir": str(tmp_path / "runs")},
    )  # type: ignore[arg-type]

    run_dir = resolve_run_dir(cfg)
    store = ArtifactStore(run_dir)

    frames = [
        FrameRecord(
            source_id="synthetic",
            frame_index=i,
            pts=i,
            time_base_num=1,
            time_base_den=1_000_000,
            pts_ns=i * 1_000_000,
            coded_width=640,
            coded_height=480,
            display_width=640,
            display_height=480,
            sampled=True,
            processed=True,
        )
        for i in range(3)
    ]
    observation = _face_observation("OBS000001", 0)
    embedding = np.array([np.random.rand(512).astype(np.float32)])
    embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)

    manifest = _manifest(video_path, video_sha256)
    store.write_manifest(manifest)
    store.write_frames(frames)
    store.write_observations([observation])
    store.write_embeddings(embedding)
    store.write_checksums()

    overlay = {
        "frame_index": 0,
        "faces": [
            {
                "observation_id": observation.observation_id,
                "pts_ns": observation.pts_ns,
                "bbox_xyxy": observation.bbox_xyxy.to_list(),
                "raw_tracklet_id": "RT000001",
                "canonical_track_id": "CT000001",
                "display_label": None,
                "detector_score": observation.detector_score,
                "quality_score": observation.quality.composite_quality_score,
                "tracking_eligible": True,
                "recognition_eligible": True,
                "rejection_reasons": [],
            }
        ],
    }
    (run_dir / "visual").mkdir(parents=True, exist_ok=True)
    (run_dir / "visual" / "overlay.jsonl").write_text(
        json.dumps(overlay) + "\n", encoding="utf-8"
    )
    tracklet = {
        "raw_tracklet_id": "RT000001",
        "strategy": "byte_iou",
        "first_frame_index": 0,
        "last_frame_index": 0,
        "first_pts_ns": 0,
        "last_pts_ns": 0,
        "observation_count": 1,
        "detection_ordinal_ids": ["OBS000001"],
        "state": "removed",
    }
    (run_dir / "replay" / "byte_iou").mkdir(parents=True, exist_ok=True)
    (run_dir / "replay" / "byte_iou" / "tracklets.jsonl").write_text(
        json.dumps(tracklet) + "\n", encoding="utf-8"
    )
    (run_dir / "ground_truth.yaml").write_text(
        yaml.safe_dump(build_ground_truth_template(video_sha256).model_dump(mode="json"))
    )

    return cfg, run_dir


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    cfg, _ = _write_run_dir(tmp_path)
    app = create_app(cfg, strategy="byte_iou")
    return TestClient(app)


def test_health(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_config(client: TestClient) -> None:
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert data["display_width"] == 640
    assert data["strategy"] == "byte_iou"
    assert "video_sha256" in data


def test_frames(client: TestClient) -> None:
    response = client.get("/api/frames")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    assert [f["frame_index"] for f in data["frames"]] == [0, 1, 2]


def test_overlay(client: TestClient) -> None:
    response = client.get("/api/overlay")
    assert response.status_code == 200
    data = response.json()
    assert data["by_frame"]["0"][0]["observation_id"] == "OBS000001"
    assert data["by_frame"]["0"][0]["recognition_eligible"] is True


def test_tracklets(client: TestClient) -> None:
    response = client.get("/api/tracklets")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["raw_tracklet_id"] == "RT000001"


def test_gt_read_and_save(client: TestClient) -> None:
    read_response = client.get("/api/gt")
    assert read_response.status_code == 200
    gt = read_response.json()
    assert gt["anchors"]

    new_anchors = [
        {
            "anchor_id": "a1",
            "label": "Rachel",
            "split": "calibration",
            "frame_index": 0,
            "observation_id": "OBS000001",
        }
    ]
    save_response = client.post("/api/gt", json={"anchors": new_anchors})
    assert save_response.status_code == 200
    assert save_response.json()["anchor_count"] == 1

    read_again = client.get("/api/gt")
    assert read_again.json()["anchors"][0]["label"] == "Rachel"


def test_video_range(client: TestClient) -> None:
    response = client.get("/api/video", headers={"Range": "bytes=0-5"})
    assert response.status_code == 206
    assert response.headers["content-range"].endswith("/19")
    assert response.content == b"dummy "

"""Unit tests for ground-truth anchor loading and resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mergenvision_video_lab.contracts import (
    BBoxXYXY,
    FaceObservation,
    GroundTruth,
    GroundTruthAnchor,
    Landmarks5,
    QualityMetrics,
)
from mergenvision_video_lab.errors import ConfigError
from mergenvision_video_lab.ground_truth import (
    build_ground_truth_template,
    load_ground_truth,
    resolve_anchor_observations,
)


def _obs(observation_id: str, frame_index: int, quality_score: float = 0.9) -> FaceObservation:
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
            composite_quality_score=quality_score,
        ),
        tracking_eligible=True,
        recognition_eligible=True,
        embedding_index=None,
    )


def test_load_ground_truth(tmp_path: Path) -> None:
    """Loading a valid YAML ground-truth file works."""
    path = tmp_path / "gt.yaml"
    data = {
        "schema_version": "mv-video-ground-truth/v1",
        "video_sha256": "abc" * 10,
        "anchors": [
            {
                "anchor_id": "rachel_early",
                "label": "Rachel",
                "split": "calibration",
                "frame_index": 10,
            }
        ],
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    gt = load_ground_truth(path)
    assert isinstance(gt, GroundTruth)
    assert gt.video_sha256 == data["video_sha256"]
    assert len(gt.anchors) == 1
    assert gt.anchors[0].anchor_id == "rachel_early"


def test_load_missing_ground_truth_raises() -> None:
    """Missing ground-truth file raises ConfigError."""
    with pytest.raises(ConfigError):
        load_ground_truth(Path("/nonexistent/gt.yaml"))


def test_resolve_anchor_by_observation_id() -> None:
    """Anchors with observation_id resolve to the exact observation."""
    observations = [_obs("obs:10", 10), _obs("obs:20", 20)]
    gt = GroundTruth(
        video_sha256="x",
        anchors=[
            GroundTruthAnchor(
                anchor_id="rachel_early",
                label="Rachel",
                split="calibration",
                observation_id="obs:10",
            )
        ],
    )

    resolved = resolve_anchor_observations(gt, observations)
    assert resolved["rachel_early"]["resolved"] is True
    assert resolved["rachel_early"]["observation"].observation_id == "obs:10"


def test_resolve_anchor_by_frame_index_prefers_quality() -> None:
    """Frame-index anchors resolve to the highest-quality face."""
    observations = [_obs("obs:low", 10, 0.3), _obs("obs:high", 10, 0.95)]
    gt = GroundTruth(
        video_sha256="x",
        anchors=[
            GroundTruthAnchor(
                anchor_id="rachel_early",
                label="Rachel",
                split="calibration",
                frame_index=10,
            )
        ],
    )

    resolved = resolve_anchor_observations(gt, observations)
    assert resolved["rachel_early"]["resolved"] is True
    assert resolved["rachel_early"]["observation"].observation_id == "obs:high"


def test_build_ground_truth_template_has_cal_holdout() -> None:
    """The default template contains calibration and holdout anchors."""
    gt = build_ground_truth_template("sha")
    splits = {a.split for a in gt.anchors}
    assert "calibration" in splits
    assert "holdout" in splits

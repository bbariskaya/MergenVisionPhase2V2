"""Unit tests for tracklet template aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from mergenvision_video_lab.contracts import BBoxXYXY, FaceObservation, Landmarks5, QualityMetrics
from mergenvision_video_lab.tracklet_templates import build_tracklet_template

LANDMARKS = Landmarks5(
    left_eye=(40.0, 50.0),
    right_eye=(70.0, 50.0),
    nose=(55.0, 70.0),
    left_mouth=(45.0, 90.0),
    right_mouth=(65.0, 90.0),
)


def _obs(
    observation_id: str,
    pts_ns: int,
    quality_score: float,
    embedding: np.ndarray,
    embedding_index: int,
) -> FaceObservation:
    return FaceObservation(
        observation_id=observation_id,
        source_id="synthetic",
        frame_index=pts_ns // 33_000_000,
        pts=pts_ns // 1_000_000,
        time_base_num=1,
        time_base_den=1_000_000,
        pts_ns=pts_ns,
        frame_width=640,
        frame_height=480,
        detection_ordinal=0,
        bbox_xyxy=BBoxXYXY(x1=100.0, y1=100.0, x2=150.0, y2=150.0),
        detector_score=0.8,
        landmarks_5=LANDMARKS,
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
        embedding_index=embedding_index,
    )


@pytest.fixture
def template_config() -> dict:
    return {
        "max_selected_samples": 5,
        "min_selected_samples": 1,
        "min_temporal_separation_ns": 200_000_000,
        "outlier_mad_scale": 3.0,
        "outlier_absolute_cosine_floor": 0.20,
        "min_quality_score": 0.2,
    }


def test_template_is_quality_weighted_centroid(template_config: dict) -> None:
    """A single high-quality observation produces a unit-normalized template."""
    emb = np.ones(512, dtype=np.float32)
    emb = emb / np.linalg.norm(emb)
    embeddings = emb.reshape(1, 512)
    obs = _obs("obs:0", 0, 0.9, emb, 0)

    result = build_tracklet_template("RT000001", [obs], embeddings, template_config)

    assert result.template is not None
    assert result.selected_count == 1
    assert result.failure_reason is None
    assert pytest.approx(float(np.linalg.norm(result.template)), rel=1e-5) == 1.0


def test_temporal_diversity_selects_spaced_observations(template_config: dict) -> None:
    """Best-quality samples from each temporal bin are selected."""
    embeddings = np.zeros((5, 512), dtype=np.float32)
    for i in range(5):
        emb = np.ones(512, dtype=np.float32) * (0.9 + i * 0.01)
        embeddings[i] = emb / np.linalg.norm(emb)

    observations = [
        _obs(f"obs:{i}", i * 250_000_000, 0.5 + i * 0.05, embeddings[i], i) for i in range(5)
    ]

    result = build_tracklet_template("RT000001", observations, embeddings, template_config)

    assert result.template is not None
    # With 5 bins across the span, each observation falls in a separate bin.
    assert result.selected_count == 5
    assert len(set(result.selected_observation_ids)) == result.selected_count


def test_blur_rejection_excludes_low_quality(template_config: dict) -> None:
    """Recognition-eligible but low-quality observations may be rejected."""
    embeddings = np.zeros((3, 512), dtype=np.float32)
    for i in range(3):
        embeddings[i] = np.ones(512, dtype=np.float32) / np.sqrt(512)

    observations = [
        _obs("obs:good:0", 0, 0.9, embeddings[0], 0),
        _obs("obs:blur:1", 250_000_000, 0.1, embeddings[1], 1),
        _obs("obs:good:2", 500_000_000, 0.95, embeddings[2], 2),
    ]

    result = build_tracklet_template("RT000001", observations, embeddings, template_config)

    assert result.template is not None
    assert "obs:blur:1" in result.rejected_observation_ids


def test_outlier_rejection_by_mad(template_config: dict) -> None:
    """One observation with very different embedding is rejected as outlier."""
    embeddings = np.zeros((4, 512), dtype=np.float32)
    for i in range(3):
        embeddings[i] = np.ones(512, dtype=np.float32) / np.sqrt(512)
    # Outlier points in the opposite direction.
    embeddings[3] = -np.ones(512, dtype=np.float32) / np.sqrt(512)

    observations = [_obs(f"obs:{i}", i * 250_000_000, 0.9, embeddings[i], i) for i in range(4)]

    result = build_tracklet_template("RT000001", observations, embeddings, template_config)

    assert result.template is not None
    assert "obs:3" in result.rejected_observation_ids


def test_insufficient_candidates_fails(template_config: dict) -> None:
    """No recognition-eligible observations yields a failed template."""
    embeddings = np.zeros((0, 512), dtype=np.float32)
    result = build_tracklet_template("RT000001", [], embeddings, template_config)
    assert result.template is None
    assert result.failure_reason == "insufficient_recognition_eligible_observations"


def test_non_unit_embedding_is_rejected(template_config: dict) -> None:
    """Embeddings that are not L2-normalized are skipped."""
    emb = np.ones(512, dtype=np.float32) * 2.0
    embeddings = emb.reshape(1, 512)
    obs = _obs("obs:0", 0, 0.9, emb, 0)

    result = build_tracklet_template("RT000001", [obs], embeddings, template_config)
    assert result.template is None


def test_tracklet_template_to_dict_round_trip(template_config: dict) -> None:
    """Serialization preserves the template vector and metadata."""
    emb = np.ones(512, dtype=np.float32) / np.sqrt(512)
    embeddings = emb.reshape(1, 512)
    obs = _obs("obs:0", 0, 0.9, emb, 0)

    result = build_tracklet_template("RT000001", [obs], embeddings, template_config)
    d = result.to_dict()

    assert d["raw_tracklet_id"] == "RT000001"
    assert d["template"] is not None
    assert len(d["template"]) == 512

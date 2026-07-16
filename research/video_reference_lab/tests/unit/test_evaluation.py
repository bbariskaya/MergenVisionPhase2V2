"""Unit tests for evaluation metrics.

These tests verify that identity evaluation compares observations through the
observation -> raw tracklet -> canonical cluster mapping, not by mixing
observation IDs with tracklet IDs.
"""

from __future__ import annotations

from mergenvision_video_lab.contracts import (
    BBoxXYXY,
    CanonicalTrack,
    FaceObservation,
    GroundTruthAnchor,
    Landmarks5,
    QualityMetrics,
)
from mergenvision_video_lab.evaluation import evaluate_gallery, evaluate_identity, evaluate_tracking


def _obs(observation_id: str, frame_index: int) -> FaceObservation:
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
        embedding_index=None,
    )


def test_evaluate_tracking_counts() -> None:
    """Tracking metrics count assigned/unassigned observations and raw tracklets."""
    observations = [_obs("a", 0), _obs("b", 1), _obs("c", 2)]
    assignments = [
        {"observation_id": "a", "frame_index": 0, "raw_tracklet_id": "RT000001"},
        {"observation_id": "b", "frame_index": 1, "raw_tracklet_id": "RT000002"},
        {"observation_id": "c", "frame_index": 2, "raw_tracklet_id": "RT000002"},
    ]

    metrics = evaluate_tracking(observations, assignments)

    assert metrics["total_observations"] == 3
    assert metrics["assigned_observations"] == 3
    assert metrics["unassigned_observations"] == 0
    assert metrics["raw_tracklet_count"] == 2
    assert metrics["duplicate_id_frames"] == 0


def test_evaluate_tracking_detects_duplicate_track_ids() -> None:
    """Same raw tracklet ID twice in one frame is flagged."""
    observations = [_obs("a", 0), _obs("b", 0)]
    assignments = [
        {"observation_id": "a", "frame_index": 0, "raw_tracklet_id": "RT000001"},
        {"observation_id": "b", "frame_index": 0, "raw_tracklet_id": "RT000001"},
    ]

    metrics = evaluate_tracking(observations, assignments)

    assert metrics["duplicate_id_frames"] == 1


def test_evaluate_identity_without_ground_truth() -> None:
    """Identity metrics are structural only when no ground truth exists."""
    clusters = [["RT000001"], ["RT000002", "RT000003"]]
    metrics = evaluate_identity(clusters, [], {}, None, None)

    assert metrics["canonical_cluster_count"] == 2
    assert metrics["ground_truth_available"] is False
    assert metrics["pairwise_f1"] is None


def test_evaluate_identity_maps_observation_through_tracklet_to_canonical() -> None:
    """Pairwise precision/recall use observation -> tracklet -> canonical mapping.

    Cluster membership is defined by raw tracklet IDs.  Ground-truth anchors
    label observations.  The evaluator must map each observation to its
    tracklet and then to its canonical cluster before comparing labels.
    """
    # Two canonical clusters: CT1 = RT1, CT2 = RT2 + RT3.
    clusters = [["RT000001"], ["RT000002", "RT000003"]]
    canonical_map = {
        "RT000001": "CT000001",
        "RT000002": "CT000002",
        "RT000003": "CT000002",
    }
    # obs_a -> RT1 (CT1, Rachel)
    # obs_b -> RT2 (CT2, Rachel)
    # obs_c -> RT3 (CT2, Monica)
    assignments = [
        {"observation_id": "obs_a", "frame_index": 0, "raw_tracklet_id": "RT000001"},
        {"observation_id": "obs_b", "frame_index": 1, "raw_tracklet_id": "RT000002"},
        {"observation_id": "obs_c", "frame_index": 2, "raw_tracklet_id": "RT000003"},
    ]
    resolved = {
        "anchor_a": {
            "anchor": GroundTruthAnchor(
                anchor_id="anchor_a", label="Rachel", split="calibration", observation_id="obs_a"
            ),
            "observation": _obs("obs_a", 0),
        },
        "anchor_b": {
            "anchor": GroundTruthAnchor(
                anchor_id="anchor_b", label="Rachel", split="holdout", observation_id="obs_b"
            ),
            "observation": _obs("obs_b", 1),
        },
        "anchor_c": {
            "anchor": GroundTruthAnchor(
                anchor_id="anchor_c", label="Monica", split="holdout", observation_id="obs_c"
            ),
            "observation": _obs("obs_c", 2),
        },
    }

    metrics = evaluate_identity(clusters, assignments, canonical_map, {}, resolved)

    # Same-label pairs: (a,b) -> different clusters => false negative.
    # Different-label pairs: (a,c) -> different clusters => true negative;
    #                       (b,c) -> same cluster => false positive.
    assert metrics["ground_truth_available"] is True
    assert metrics["pairwise_precision"] == 0.0
    assert metrics["pairwise_recall"] == 0.0
    assert metrics["pairwise_f1"] == 0.0
    assert metrics["cluster_purity"] == 0.5


def test_evaluate_identity_perfect_clustering() -> None:
    """When same-label anchors share a canonical cluster metrics are perfect."""
    clusters = [["RT000001", "RT000002"], ["RT000003"]]
    canonical_map = {
        "RT000001": "CT000001",
        "RT000002": "CT000001",
        "RT000003": "CT000002",
    }
    assignments = [
        {"observation_id": "obs_a", "frame_index": 0, "raw_tracklet_id": "RT000001"},
        {"observation_id": "obs_b", "frame_index": 1, "raw_tracklet_id": "RT000002"},
        {"observation_id": "obs_c", "frame_index": 2, "raw_tracklet_id": "RT000003"},
    ]
    resolved = {
        "anchor_a": {
            "anchor": GroundTruthAnchor(
                anchor_id="anchor_a", label="Rachel", split="calibration", observation_id="obs_a"
            ),
            "observation": _obs("obs_a", 0),
        },
        "anchor_b": {
            "anchor": GroundTruthAnchor(
                anchor_id="anchor_b", label="Rachel", split="holdout", observation_id="obs_b"
            ),
            "observation": _obs("obs_b", 1),
        },
        "anchor_c": {
            "anchor": GroundTruthAnchor(
                anchor_id="anchor_c", label="Monica", split="holdout", observation_id="obs_c"
            ),
            "observation": _obs("obs_c", 2),
        },
    }

    metrics = evaluate_identity(clusters, assignments, canonical_map, {}, resolved)

    assert metrics["pairwise_precision"] == 1.0
    assert metrics["pairwise_recall"] == 1.0
    assert metrics["pairwise_f1"] == 1.0
    assert metrics["cluster_purity"] == 1.0


def test_evaluate_gallery_known_unknown() -> None:
    """Gallery summary distinguishes known and unknown by the actual decision.

    A track with a top-1 candidate label but ``decision_reason != "gallery_match"``
    must be counted as unknown.
    """
    known_track = CanonicalTrack(
        canonical_track_id="CT000001",
        raw_tracklet_ids=["RT000001"],
        display_label="Rachel",
        first_seen_pts_ns=0,
        last_seen_pts_ns=33_000_000,
        total_duration_ns=33_000_000,
        appearances=[],
        detections=[],
        template_evidence={},
        gallery_top1_label="Rachel",
        gallery_top1_cosine=0.72,
        decision_reason="gallery_match",
        confidence_evidence={},
    )
    unknown_track = CanonicalTrack(
        canonical_track_id="CT000002",
        raw_tracklet_ids=["RT000002"],
        display_label=None,
        first_seen_pts_ns=100_000_000,
        last_seen_pts_ns=133_000_000,
        total_duration_ns=33_000_000,
        appearances=[],
        detections=[],
        template_evidence={},
        gallery_top1_label="Rachel",  # diagnostic only
        gallery_top1_cosine=0.48,
        decision_reason="gallery_rejected",
        confidence_evidence={},
    )

    metrics = evaluate_gallery([known_track, unknown_track])

    assert metrics["known_count"] == 1
    assert metrics["unknown_count"] == 1
    assert metrics["total_tracks"] == 2

"""Unit tests for canonical track aggregation."""

from __future__ import annotations

import numpy as np

from mergenvision_video_lab.aggregation import aggregate_canonical_tracks, build_canonical_track
from mergenvision_video_lab.contracts import (
    BBoxXYXY,
    CanonicalTrack,
    FaceObservation,
    Landmarks5,
    QualityMetrics,
)

LANDMARKS = Landmarks5(
    left_eye=(40.0, 50.0),
    right_eye=(70.0, 50.0),
    nose=(55.0, 70.0),
    left_mouth=(45.0, 90.0),
    right_mouth=(65.0, 90.0),
)


def _obs(
    observation_id: str,
    frame_index: int,
    pts_ns: int,
) -> FaceObservation:
    return FaceObservation(
        observation_id=observation_id,
        source_id="synthetic",
        frame_index=frame_index,
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
            composite_quality_score=0.9,
        ),
        tracking_eligible=True,
        recognition_eligible=True,
        embedding_index=None,
    )


def _emb(value: float) -> np.ndarray:
    v = np.ones(512, dtype=np.float32) * value
    return v / np.linalg.norm(v)


def test_aggregate_canonical_tracks_groups_observations() -> None:
    """Observations are routed to the correct canonical cluster."""
    observations = [
        _obs("obs:a:0", 0, 0),
        _obs("obs:a:1", 1, 33_000_000),
        _obs("obs:b:2", 5, 165_000_000),
    ]
    assignments = [
        {"observation_id": "obs:a:0", "raw_tracklet_id": "RT000001"},
        {"observation_id": "obs:a:1", "raw_tracklet_id": "RT000001"},
        {"observation_id": "obs:b:2", "raw_tracklet_id": "RT000002"},
    ]
    clusters = [["RT000001"], ["RT000002"]]
    cluster_ids = ["CT000001", "CT000002"]
    templates = {"RT000001": _emb(1.0), "RT000002": _emb(-1.0)}

    tracks = aggregate_canonical_tracks(
        observations,
        assignments,
        clusters,
        cluster_ids,
        templates,
        max_gap_multiplier=2.5,
    )

    assert len(tracks) == 2
    by_id = {t.canonical_track_id: t for t in tracks}
    assert len(by_id["CT000001"].detections) == 2
    assert len(by_id["CT000002"].detections) == 1
    assert by_id["CT000001"].display_label is None


def test_known_gallery_match_sets_display_label() -> None:
    """Only known gallery matches may set display_label."""
    observations = [_obs("obs:a:0", 0, 0)]
    gallery_match = {
        "known": True,
        "top1_label": "Rachel",
        "top1_cosine": 0.72,
        "top2_label": "Monica",
        "top2_cosine": 0.55,
        "margin": 0.17,
        "threshold": 0.45,
        "margin_threshold": 0.05,
    }

    track = build_canonical_track(
        canonical_track_id="CT000001",
        raw_tracklet_ids=["RT000001"],
        observations=observations,
        member_templates=[_emb(1.0)],
        gallery_match=gallery_match,
        max_gap_multiplier=2.5,
    )

    assert isinstance(track, CanonicalTrack)
    assert track.display_label == "Rachel"
    assert track.decision_reason == "gallery_match"


def test_unknown_gallery_match_hides_top1_label() -> None:
    """Rejected matches must not expose top1 as display_label."""
    observations = [_obs("obs:a:0", 0, 0)]
    gallery_match = {
        "known": False,
        "top1_label": "Rachel",
        "top1_cosine": 0.40,
        "top2_label": "Monica",
        "top2_cosine": 0.35,
        "margin": 0.05,
        "threshold": 0.45,
        "margin_threshold": 0.05,
    }

    track = build_canonical_track(
        canonical_track_id="CT000001",
        raw_tracklet_ids=["RT000001"],
        observations=observations,
        member_templates=[_emb(1.0)],
        gallery_match=gallery_match,
        max_gap_multiplier=2.5,
    )

    assert track.display_label is None
    assert track.gallery_top1_label == "Rachel"
    assert track.decision_reason == "complete_link_reconciliation"


def test_empty_cluster_has_zero_duration() -> None:
    """A cluster with no observations has zero duration and empty detections."""
    tracks = aggregate_canonical_tracks(
        observations=[],
        assignments=[],
        clusters=[["RT000001"]],
        cluster_ids=["CT000001"],
        templates={"RT000001": _emb(1.0)},
        max_gap_multiplier=2.5,
    )

    assert len(tracks) == 1
    assert tracks[0].total_duration_ns == 0
    assert tracks[0].detections == []

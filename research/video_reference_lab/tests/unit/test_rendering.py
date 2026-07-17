"""Unit tests for visual diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from mergenvision_video_lab.contracts import BBoxXYXY, FaceObservation, Landmarks5, QualityMetrics
from mergenvision_video_lab.visualization import make_overlay_jsonl, make_quality_histograms


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


def test_overlay_jsonl_writes_per_frame_records(tmp_path: Path) -> None:
    """Overlay JSONL groups observations by frame with canonical/label info."""
    observations = [_obs("a", 0), _obs("b", 0), _obs("c", 1)]
    assignments = [
        {"observation_id": "a", "raw_tracklet_id": "RT000001"},
        {"observation_id": "b", "raw_tracklet_id": "RT000002"},
        {"observation_id": "c", "raw_tracklet_id": "RT000001"},
    ]
    canonical_map = {"RT000001": "CT000001", "RT000002": "CT000002"}
    labels = {"CT000001": "Rachel", "CT000002": None}
    output = tmp_path / "overlay.jsonl"

    make_overlay_jsonl(observations, assignments, canonical_map, labels, output)

    assert output.exists()
    records = []
    with open(output, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    assert len(records) == 2
    by_frame = {r["frame_index"]: r["faces"] for r in records}
    assert len(by_frame[0]) == 2
    assert len(by_frame[1]) == 1
    # Unknown track must not display a label.
    frame0_unknown = [f for f in by_frame[0] if f["raw_tracklet_id"] == "RT000002"][0]
    assert frame0_unknown["display_label"] is None


def test_overlay_jsonl_includes_quality_and_eligibility(tmp_path: Path) -> None:
    """Overlay records expose observation_id, eligibility and rejection reasons."""
    obs = _obs("a", 0)
    obs.rejection_reasons = ["blur"]
    obs.recognition_eligible = False
    assignments = [{"observation_id": "a", "raw_tracklet_id": "RT000001"}]
    output = tmp_path / "overlay.jsonl"

    make_overlay_jsonl([obs], assignments, {}, {}, output)

    with open(output, encoding="utf-8") as f:
        record = json.loads(f.readline())

    face = record["faces"][0]
    assert face["observation_id"] == "a"
    assert face["recognition_eligible"] is False
    assert "blur" in face["rejection_reasons"]
    assert face["tracking_eligible"] is True


def test_quality_histograms_runs_with_no_observations(tmp_path: Path) -> None:
    """Quality histograms handle empty observation list gracefully."""
    output = tmp_path / "quality.jpg"
    make_quality_histograms([], output)
    assert output.exists()

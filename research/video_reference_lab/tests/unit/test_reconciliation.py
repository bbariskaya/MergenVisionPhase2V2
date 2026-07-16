"""Unit tests for offline tracklet reconciliation."""

from __future__ import annotations

import numpy as np
import pytest

from mergenvision_video_lab.reconciliation import (
    TrackletRecord,
    build_cannot_link_pairs,
    reconcile_tracklets,
)


def _template(angle_deg: float) -> np.ndarray:
    """Deterministic unit embedding pointing at ``angle_deg`` in the first two
    dimensions. Cosine between two such embeddings is ``cos(angle_a - angle_b)``.
    """
    angle = np.deg2rad(angle_deg)
    emb = np.zeros(512, dtype=np.float32)
    emb[0] = np.cos(angle)
    emb[1] = np.sin(angle)
    return emb


def _record(
    raw_tracklet_id: str,
    frame_indices: list[int],
    template_value: float,
) -> TrackletRecord:
    return TrackletRecord(
        raw_tracklet_id=raw_tracklet_id,
        strategy="byte_iou",
        observation_ids=[f"{raw_tracklet_id}:{f}" for f in frame_indices],
        frame_indices=frame_indices,
        first_pts_ns=frame_indices[0] * 33_000_000,
        last_pts_ns=frame_indices[-1] * 33_000_000,
        template=_template(template_value),
    )


@pytest.fixture
def reconciliation_config() -> dict:
    return {
        "min_tracklet_cosine": 0.60,
        "min_cluster_member_cosine": 0.55,
        "min_top1_top2_margin": 0.05,
        "overlap_tolerance_ns": 0,
    }


def test_same_frame_co_occurrence_is_cannot_link() -> None:
    """Two tracklets sharing a frame cannot be the same person."""
    a = _record("RT000001", [0, 1, 2], 1.0)
    b = _record("RT000002", [2, 3, 4], 1.0)
    pairs = build_cannot_link_pairs([a, b])
    assert ("RT000001", "RT000002") in pairs


def test_interval_overlap_without_co_occurrence_is_not_cannot_link() -> None:
    """Mere temporal overlap is insufficient for cannot-link."""
    a = _record("RT000001", [0, 1], 1.0)
    b = _record("RT000002", [2, 3], 1.0)
    pairs = build_cannot_link_pairs([a, b])
    assert ("RT000001", "RT000002") not in pairs


def test_human_cannot_link_is_respected() -> None:
    """Explicit human-declared pairs are absolute cannot-links."""
    a = _record("RT000001", [0, 1], 1.0)
    b = _record("RT000002", [5, 6], 1.0)
    human = {("RT000001", "RT000002")}
    pairs = build_cannot_link_pairs([a, b], human_cannot_links=human)
    assert ("RT000001", "RT000002") in pairs


def test_complete_link_rejects_transitive_chain(reconciliation_config: dict) -> None:
    """A~B and B~C with weak A~C must not merge all three."""
    # a-b and b-c are just above the threshold; a-c falls well below it.
    a = _record("RT000001", [0, 1, 2], 50.0)
    b = _record("RT000002", [10, 11, 12], 0.0)
    c = _record("RT000003", [20, 21, 22], -50.0)

    result = reconcile_tracklets([a, b, c], reconciliation_config)

    clusters = result["clusters"]
    # A and B may merge, C must stay separate.
    assert any(len(cluster) == 2 for cluster in clusters)
    assert any(len(cluster) == 1 for cluster in clusters)
    assert result["transitive_rejection_count"] >= 1


def test_cannot_link_overrides_high_cosine(reconciliation_config: dict) -> None:
    """High cosine cannot merge tracklets with a same-frame cannot-link."""
    a = _record("RT000001", [0, 1, 2], 1.0)
    b = _record("RT000002", [2, 3, 4], 1.0)

    result = reconcile_tracklets([a, b], reconciliation_config)

    clusters = result["clusters"]
    assert len(clusters) == 2
    assert all(len(c) == 1 for c in clusters)


def test_canonical_ids_are_deterministic(reconciliation_config: dict) -> None:
    """Canonical IDs sort by earliest PTS and use zero-padded format."""
    a = _record("RT000001", [10, 11], 90.0)
    b = _record("RT000002", [0, 1], 0.0)
    c = _record("RT000003", [20, 21], 180.0)

    result = reconcile_tracklets([a, b, c], reconciliation_config)

    assert result["canonical_track_ids"] == ["CT000001", "CT000002", "CT000003"]
    # Earliest PTS cluster is first.
    assert result["canonical_map"]["RT000002"] == "CT000001"


def test_empty_tracklet_list(reconciliation_config: dict) -> None:
    """Empty input yields no clusters."""
    result = reconcile_tracklets([], reconciliation_config)
    assert result["clusters"] == []
    assert result["canonical_track_ids"] == []

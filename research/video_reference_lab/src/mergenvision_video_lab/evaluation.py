"""Evaluation metrics for tracking, identity reconciliation, and gallery.

All identity comparisons use the correct namespace mapping:

    observation_id -> raw_tracklet_id -> canonical_track_id

Never compare observation IDs directly with raw tracklet IDs or canonical
cluster member lists.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from mergenvision_video_lab.contracts import CanonicalTrack, FaceObservation


def evaluate_tracking(
    observations: list[FaceObservation],
    assignments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute tracking consistency metrics."""
    obs_by_frame: dict[int, list[str]] = defaultdict(list)
    for obs in observations:
        obs_by_frame[obs.frame_index].append(obs.observation_id)

    assignment_by_obs = {a["observation_id"]: a for a in assignments}
    frame_track_ids: dict[int, list[str]] = defaultdict(list)
    for a in assignments:
        frame_track_ids[a["frame_index"]].append(a["raw_tracklet_id"])

    duplicate_id_frames = 0
    for _frame_idx, track_ids in frame_track_ids.items():
        if len(track_ids) != len(set(track_ids)):
            duplicate_id_frames += 1

    assigned_count = len([obs for obs in observations if obs.observation_id in assignment_by_obs])

    return {
        "total_observations": len(observations),
        "assigned_observations": assigned_count,
        "unassigned_observations": len(observations) - assigned_count,
        "duplicate_id_frames": duplicate_id_frames,
        "raw_tracklet_count": len({a["raw_tracklet_id"] for a in assignments}),
    }


def _build_observation_to_canonical_map(
    assignments: list[dict[str, Any]],
    canonical_map: dict[str, str],
) -> dict[str, str | None]:
    """Map each assigned observation to its canonical track ID, if any."""
    obs_to_tracklet: dict[str, str] = {}
    for assignment in assignments:
        obs_to_tracklet[assignment["observation_id"]] = assignment["raw_tracklet_id"]

    result: dict[str, str | None] = {}
    for obs_id, tracklet_id in obs_to_tracklet.items():
        result[obs_id] = canonical_map.get(tracklet_id)
    return result


def evaluate_identity(
    clusters: list[list[str]],
    assignments: list[dict[str, Any]] | None,
    canonical_map: dict[str, str] | None,
    ground_truth: Any | None = None,
    resolved_anchors: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute identity/reconciliation metrics.

    Parameters
    ----------
    clusters:
        Lists of raw tracklet IDs, one list per canonical cluster.
    assignments:
        Observation -> raw tracklet assignments from replay.
    canonical_map:
        Raw tracklet ID -> canonical track ID mapping.
    ground_truth:
        Optional ground-truth object (currently unused structurally; anchors
        drive labeled evaluation).
    resolved_anchors:
        Optional resolved ground-truth anchors as returned by
        ``resolve_anchor_observations``.
    """
    result: dict[str, Any] = {
        "canonical_cluster_count": len(clusters),
        "cluster_size_distribution": [len(c) for c in clusters],
        "ground_truth_available": resolved_anchors is not None,
        "labeled_anchor_count": 0,
        "resolved_anchor_count": 0,
        "same_label_pair_recall": None,
        "different_label_pair_rejection": None,
        "pairwise_precision": None,
        "pairwise_recall": None,
        "pairwise_f1": None,
        "cluster_purity": None,
        "uncertain_pair_count": 0,
        "transitive_chain_rejection_count": None,
    }

    if assignments is None or canonical_map is None or resolved_anchors is None:
        return result

    obs_to_canonical = _build_observation_to_canonical_map(assignments, canonical_map)

    # Build canonical_id -> set of raw tracklet IDs for fast lookup.
    canonical_to_tracklets: dict[str, set[str]] = {}
    for cluster in clusters:
        canonical_id = canonical_map.get(cluster[0]) if cluster else None
        if canonical_id is not None:
            canonical_to_tracklets[canonical_id] = set(cluster)

    # Map each resolved anchor to its canonical cluster.
    label_by_canonical: dict[str, str] = {}
    canonical_by_anchor: dict[str, str | None] = {}
    for anchor_id, data in resolved_anchors.items():
        obs = data.get("observation")
        if obs is None:
            canonical_by_anchor[anchor_id] = None
            continue
        canonical_id = obs_to_canonical.get(obs.observation_id)
        canonical_by_anchor[anchor_id] = canonical_id
        if canonical_id is not None:
            anchor_label = data["anchor"].label
            existing = label_by_canonical.get(canonical_id)
            if existing is not None and existing != anchor_label:
                # A canonical cluster contains anchors with different labels:
                # this is a purity violation.
                label_by_canonical[canonical_id] = f"CONFLICT:{existing}|{anchor_label}"
            else:
                label_by_canonical[canonical_id] = anchor_label

    resolved = [a for a, cid in canonical_by_anchor.items() if cid is not None]
    result["resolved_anchor_count"] = len(resolved)
    result["labeled_anchor_count"] = len(resolved_anchors)

    if len(resolved) < 2:
        return result

    # Pairwise metrics over resolved labeled anchors.
    true_same = 0
    pred_same = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    anchor_ids = list(resolved)
    for i, a_id in enumerate(anchor_ids):
        a_canonical = canonical_by_anchor[a_id]
        a_label = resolved_anchors[a_id]["anchor"].label
        for b_id in anchor_ids[i + 1 :]:
            b_canonical = canonical_by_anchor[b_id]
            b_label = resolved_anchors[b_id]["anchor"].label
            same_label = a_label == b_label
            same_cluster = a_canonical == b_canonical and a_canonical is not None

            if same_label:
                true_same += 1
            if same_cluster:
                pred_same += 1
            if same_label and same_cluster:
                true_positive += 1
            elif same_cluster and not same_label:
                false_positive += 1
            elif same_label and not same_cluster:
                false_negative += 1

    total_pairs = len(anchor_ids) * (len(anchor_ids) - 1) // 2
    different_label_pairs = total_pairs - true_same

    precision = true_positive / pred_same if pred_same else 0.0
    recall = true_positive / true_same if true_same else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    result["same_label_pair_recall"] = recall
    result["different_label_pair_rejection"] = (
        1.0 - (false_positive / different_label_pairs) if different_label_pairs else None
    )
    result["pairwise_precision"] = precision
    result["pairwise_recall"] = recall
    result["pairwise_f1"] = f1

    # Cluster purity: fraction of clusters that contain at most one distinct label.
    if label_by_canonical:
        pure_clusters = sum(
            1 for label in label_by_canonical.values() if not label.startswith("CONFLICT:")
        )
        result["cluster_purity"] = pure_clusters / len(label_by_canonical)

    return result


def evaluate_gallery(
    canonical_tracks: list[CanonicalTrack],
) -> dict[str, Any]:
    """Summarize gallery-matching decisions across canonical tracks.

    A track counts as ``known`` only when the gallery decision actually passed
    (``decision_reason == "gallery_match"``) and ``display_label`` is set.  The
    mere presence of a top-1 candidate label is diagnostic evidence only and
    must not be reported as a known identity.
    """
    known = 0
    unknown = 0
    for track in canonical_tracks:
        if track.decision_reason == "gallery_match" and track.display_label is not None:
            known += 1
        else:
            unknown += 1
    return {
        "known_count": known,
        "unknown_count": unknown,
        "total_tracks": len(canonical_tracks),
    }

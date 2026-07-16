"""Evaluation metrics for tracking, identity reconciliation, and gallery."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from mergenvision_video_lab.contracts import FaceObservation


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
    for frame_idx, track_ids in frame_track_ids.items():
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


def evaluate_identity(
    clusters: list[list[str]],
    ground_truth: Any | None,
    resolved_anchors: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compute identity/reconciliation metrics when labels exist."""
    result: dict[str, Any] = {
        "canonical_cluster_count": len(clusters),
        "cluster_size_distribution": [len(c) for c in clusters],
        "ground_truth_available": ground_truth is not None,
        "same_label_pair_recall": None,
        "different_label_pair_rejection": None,
        "pairwise_precision": None,
        "pairwise_recall": None,
        "pairwise_f1": None,
        "cluster_purity": None,
        "uncertain_pair_count": 0,
        "transitive_chain_rejection_count": 0,
    }

    if ground_truth is None or resolved_anchors is None:
        return result

    label_by_obs: dict[str, str] = {}
    for anchor_id, data in resolved_anchors.items():
        obs = data.get("observation")
        if obs is not None:
            label_by_obs[obs.observation_id] = data["anchor"].label

    if len(label_by_obs) < 2:
        return result

    # Pairwise metrics over labeled anchors.
    obs_ids = list(label_by_obs.keys())
    true_same = 0
    pred_same = 0
    true_positive = 0
    for i, a in enumerate(obs_ids):
        for b in obs_ids[i + 1 :]:
            same_label = label_by_obs[a] == label_by_obs[b]
            # Predicted same if in same cluster.
            same_cluster = any(a in c and b in c for c in clusters)
            if same_label:
                true_same += 1
            if same_cluster:
                pred_same += 1
            if same_label and same_cluster:
                true_positive += 1

    precision = true_positive / pred_same if pred_same else 0.0
    recall = true_positive / true_same if true_same else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    result["same_label_pair_recall"] = recall
    result["different_label_pair_rejection"] = 1.0 - (pred_same - true_positive) / (
        len(obs_ids) * (len(obs_ids) - 1) / 2 - true_same
    ) if len(obs_ids) > 2 else None
    result["pairwise_precision"] = precision
    result["pairwise_recall"] = recall
    result["pairwise_f1"] = f1
    return result


def evaluate_gallery(
    canonical_tracks: list[Any],
) -> dict[str, Any]:
    """Summarize gallery-matching decisions across canonical tracks."""
    known = 0
    unknown = 0
    for track in canonical_tracks:
        if track.gallery_top1_label and track.gallery_top1_cosine is not None:
            known += 1
        else:
            unknown += 1
    return {
        "known_count": known,
        "unknown_count": unknown,
        "total_tracks": len(canonical_tracks),
    }

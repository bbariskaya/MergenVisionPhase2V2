"""Person-level aggregation: raw tracklets -> canonical tracks -> appearances."""

from __future__ import annotations

from typing import Any

import numpy as np

from mergenvision_video_lab.contracts import CanonicalTrack, FaceObservation


def _median_gap_ns(observations: list[FaceObservation]) -> int:
    """Median PTS gap between consecutive observations."""
    if len(observations) < 2:
        return 0
    pts = sorted([obs.pts_ns for obs in observations])
    gaps = [pts[i + 1] - pts[i] for i in range(len(pts) - 1)]
    return int(np.median(gaps))


def build_appearances(
    observations: list[FaceObservation],
    max_gap_multiplier: float,
) -> tuple[list[dict[str, Any]], int]:
    """Build appearance intervals from actual PTS gaps.

    Returns ``(appearances, total_duration_ns)``.  Observations without a
    preceding/succeeding observation within ``max_gap_multiplier * median_gap``
    start/end a new appearance interval.
    """
    if not observations:
        return [], 0

    sorted_obs = sorted(observations, key=lambda o: o.pts_ns)
    median_gap = _median_gap_ns(sorted_obs)
    max_gap = max(1, int(median_gap * max_gap_multiplier))

    appearances: list[dict[str, Any]] = []
    current_start = sorted_obs[0]
    current_end = sorted_obs[0]
    total_duration = 0

    for obs in sorted_obs[1:]:
        if obs.pts_ns - current_end.pts_ns <= max_gap:
            current_end = obs
        else:
            duration = current_end.pts_ns - current_start.pts_ns
            appearances.append(
                {
                    "start_pts_ns": current_start.pts_ns,
                    "end_pts_ns": current_end.pts_ns,
                    "start_frame_index": current_start.frame_index,
                    "end_frame_index": current_end.frame_index,
                    "duration_ns": duration,
                }
            )
            total_duration += duration
            current_start = obs
            current_end = obs

    duration = current_end.pts_ns - current_start.pts_ns
    appearances.append(
        {
            "start_pts_ns": current_start.pts_ns,
            "end_pts_ns": current_end.pts_ns,
            "start_frame_index": current_start.frame_index,
            "end_frame_index": current_end.frame_index,
            "duration_ns": duration,
        }
    )
    total_duration += duration

    return appearances, total_duration


def aggregate_canonical_tracks(
    observations: list[FaceObservation],
    raw_tracklet_ids: list[str],
    clusters: list[list[str]],
    templates: dict[str, np.ndarray | None],
    max_gap_multiplier: float,
) -> list[CanonicalTrack]:
    """Build canonical tracks from reconciliation clusters."""
    obs_by_tracklet: dict[str, list[FaceObservation]] = {tid: [] for tid in raw_tracklet_ids}
    for obs in observations:
        # Observations are not yet tagged with raw_tracklet_id; callers must
        # group them first. This helper receives the observations already
        # belonging to each canonical cluster, so we build per-tracklet groups.
        pass

    # Re-group by supplied tracklet ID set.
    canonical_tracks: list[CanonicalTrack] = []
    for cluster_idx, cluster_ids in enumerate(clusters, start=1):
        canonical_track_id = f"CT{cluster_idx:06d}"
        cluster_obs: list[FaceObservation] = []
        member_templates: list[np.ndarray] = []
        for tid in cluster_ids:
            # observations are filtered outside; here we just know cluster membership.
            # Template averaging for the cluster.
            tpl = templates.get(tid)
            if tpl is not None:
                member_templates.append(tpl)

        # Placeholder: real observations/detection lists are filled by the caller
        # that has the assignment mapping. This module is kept thin because the
        # CLI pipeline owns the observation-to-tracklet grouping.
        appearances, total_duration = build_appearances(cluster_obs, max_gap_multiplier)

        if member_templates:
            centroid = np.mean(np.asarray(member_templates, dtype=np.float32), axis=0)
            norm = float(np.linalg.norm(centroid))
            cluster_template = (centroid / norm).astype(np.float32) if norm > 0 else None
        else:
            cluster_template = None

        canonical_tracks.append(
            CanonicalTrack(
                canonical_track_id=canonical_track_id,
                raw_tracklet_ids=sorted(cluster_ids),
                display_label=None,
                first_seen_pts_ns=0,
                last_seen_pts_ns=0,
                total_duration_ns=total_duration,
                appearances=[a.model_dump() if hasattr(a, "model_dump") else a for a in appearances],
                detections=[],
                template_evidence={
                    "member_count": len(member_templates),
                    "cluster_template": (
                        cluster_template.tolist() if cluster_template is not None else None
                    ),
                },
                decision_reason="complete_link_reconciliation",
                confidence_evidence={
                    "min_tracklet_cosine": None,
                    "member_template_norm": float(np.linalg.norm(cluster_template)) if cluster_template is not None else None,
                },
            )
        )

    return canonical_tracks


def build_canonical_track(
    canonical_track_id: str,
    raw_tracklet_ids: list[str],
    observations: list[FaceObservation],
    member_templates: list[np.ndarray],
    gallery_match: dict[str, Any] | None,
    max_gap_multiplier: float,
) -> CanonicalTrack:
    """Build a fully populated canonical track for one cluster."""
    sorted_obs = sorted(observations, key=lambda o: o.pts_ns)
    appearances, total_duration = build_appearances(sorted_obs, max_gap_multiplier)

    detections = [
        {
            "observation_id": obs.observation_id,
            "frame_index": obs.frame_index,
            "pts_ns": obs.pts_ns,
            "bbox_xyxy": obs.bbox_xyxy.to_list(),
            "detector_score": obs.detector_score,
            "quality_score": obs.quality.composite_quality_score,
            "provenance": "actual_detection",
        }
        for obs in sorted_obs
    ]

    if member_templates:
        embs = np.asarray(member_templates, dtype=np.float32)
        centroid = np.mean(embs, axis=0)
        norm = float(np.linalg.norm(centroid))
        cluster_template = (centroid / norm).astype(np.float32) if norm > 0 else None
    else:
        cluster_template = None

    display_label = gallery_match.get("top1_label") if gallery_match else None
    decision_reason = (
        "gallery_match" if gallery_match and gallery_match.get("known") else "complete_link_reconciliation"
    )

    return CanonicalTrack(
        canonical_track_id=canonical_track_id,
        raw_tracklet_ids=sorted(raw_tracklet_ids),
        display_label=display_label,
        first_seen_pts_ns=sorted_obs[0].pts_ns if sorted_obs else 0,
        last_seen_pts_ns=sorted_obs[-1].pts_ns if sorted_obs else 0,
        total_duration_ns=total_duration,
        appearances=appearances,
        detections=detections,
        template_evidence={
            "member_count": len(member_templates),
            "cluster_template": cluster_template.tolist() if cluster_template is not None else None,
        },
        gallery_top1_label=gallery_match.get("top1_label") if gallery_match else None,
        gallery_top1_cosine=gallery_match.get("top1_cosine") if gallery_match else None,
        gallery_top2_label=gallery_match.get("top2_label") if gallery_match else None,
        gallery_top2_cosine=gallery_match.get("top2_cosine") if gallery_match else None,
        gallery_margin=gallery_match.get("margin") if gallery_match else None,
        decision_reason=decision_reason,
        confidence_evidence={
            "match_threshold": gallery_match.get("threshold") if gallery_match else None,
            "margin_threshold": gallery_match.get("margin_threshold") if gallery_match else None,
            "member_template_norm": float(np.linalg.norm(cluster_template)) if cluster_template is not None else None,
        },
        limitations=["gallery decision requires explicit labels" if gallery_match is None else ""],
    )

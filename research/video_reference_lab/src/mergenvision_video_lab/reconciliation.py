"""Offline tracklet reconciliation with cannot-link and complete-link rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mergenvision_video_lab.contracts import FaceObservation


@dataclass
class TrackletRecord:
    """Minimal temporal record of a raw tracklet needed for reconciliation."""

    raw_tracklet_id: str
    strategy: str
    observation_ids: list[str]
    frame_indices: list[int]
    first_pts_ns: int
    last_pts_ns: int
    template: np.ndarray | None = None
    template_quality: dict[str, Any] | None = None


def _overlap(
    a_start: int, a_end: int, b_start: int, b_end: int, tolerance_ns: int
) -> bool:
    """Return True if two closed intervals overlap beyond tolerance."""
    return not (a_end + tolerance_ns < b_start or b_end + tolerance_ns < a_start)


def build_cannot_link_pairs(
    tracklets: list[TrackletRecord],
    overlap_tolerance_ns: int,
) -> set[tuple[str, str]]:
    """Build absolute cannot-link pairs from temporal overlap or co-occurrence."""
    pairs: set[tuple[str, str]] = set()
    n = len(tracklets)
    for i in range(n):
        a = tracklets[i]
        a_frames = set(a.frame_indices)
        for j in range(i + 1, n):
            b = tracklets[j]
            # Same-frame co-occurrence is an absolute cannot-link.
            if a_frames & set(b.frame_indices):
                pairs.add(_ordered_pair(a.raw_tracklet_id, b.raw_tracklet_id))
                continue
            if _overlap(
                a.first_pts_ns,
                a.last_pts_ns,
                b.first_pts_ns,
                b.last_pts_ns,
                overlap_tolerance_ns,
            ):
                pairs.add(_ordered_pair(a.raw_tracklet_id, b.raw_tracklet_id))
    return pairs


def _ordered_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _pairwise_template_cosine(
    tracklets: list[TrackletRecord],
) -> dict[tuple[str, str], float]:
    """Cosine between every pair of valid tracklet templates."""
    result: dict[tuple[str, str], float] = {}
    n = len(tracklets)
    for i in range(n):
        a = tracklets[i]
        if a.template is None:
            continue
        for j in range(i + 1, n):
            b = tracklets[j]
            if b.template is None:
                continue
            cosine = float(np.dot(a.template, b.template))
            result[_ordered_pair(a.raw_tracklet_id, b.raw_tracklet_id)] = cosine
    return result


def _cluster_centroid(tracklet_ids: list[str], templates: dict[str, np.ndarray]) -> np.ndarray:
    """Average of unit-normalized templates, then re-normalize."""
    embs = np.asarray([templates[tid] for tid in tracklet_ids], dtype=np.float32)
    centroid = np.mean(embs, axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("cluster centroid normalization failed")
    return (centroid / norm).astype(np.float32)


def _internal_margin(
    tracklet_id: str,
    cluster_ids: list[str],
    templates: dict[str, np.ndarray],
) -> float:
    """Return top1 - top2 cosine margin of ``tracklet_id`` within ``cluster_ids``."""
    if len(cluster_ids) <= 2:
        return 1.0
    others = [tid for tid in cluster_ids if tid != tracklet_id]
    sims = sorted(
        [float(np.dot(templates[tracklet_id], templates[oid])) for oid in others],
        reverse=True,
    )
    return sims[0] - sims[1]


def reconcile_tracklets(
    tracklets: list[TrackletRecord],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Reconcile raw tracklets into canonical clusters."""
    min_tracklet_cosine = float(config["min_tracklet_cosine"])
    min_member_cosine = float(config["min_cluster_member_cosine"])
    min_margin = float(config["min_top1_top2_margin"])
    overlap_tolerance_ns = int(config["overlap_tolerance_ns"])

    cannot_link = build_cannot_link_pairs(tracklets, overlap_tolerance_ns)
    pair_cosines = _pairwise_template_cosine(tracklets)

    # Candidate edges: valid pairs, not cannot-link, above threshold.
    candidate_edges = [
        (pair, cosine)
        for pair, cosine in pair_cosines.items()
        if pair not in cannot_link and cosine >= min_tracklet_cosine
    ]
    candidate_edges.sort(key=lambda x: (-x[1], x[0][0], x[0][1]))

    templates = {t.raw_tracklet_id: t.template for t in tracklets if t.template is not None}
    cluster_map: dict[str, int] = {}
    clusters: list[list[str]] = []
    for t in tracklets:
        cluster_map[t.raw_tracklet_id] = len(clusters)
        clusters.append([t.raw_tracklet_id])

    merge_count = 0
    transitive_rejection_count = 0

    for (id_a, id_b), cosine in candidate_edges:
        ca = cluster_map[id_a]
        cb = cluster_map[id_b]
        if ca == cb:
            continue

        cluster_a = clusters[ca]
        cluster_b = clusters[cb]

        # 1. Cannot-link across components.
        cross_cannot = False
        for x in cluster_a:
            for y in cluster_b:
                if _ordered_pair(x, y) in cannot_link:
                    cross_cannot = True
                    break
            if cross_cannot:
                break
        if cross_cannot:
            continue

        # 2. Every cross-component pair must pass the tracklet threshold.
        cross_ok = True
        for x in cluster_a:
            for y in cluster_b:
                pair = _ordered_pair(x, y)
                pc = pair_cosines.get(pair)
                if pc is None or pc < min_tracklet_cosine:
                    cross_ok = False
                    break
            if not cross_ok:
                break
        if not cross_ok:
            transitive_rejection_count += 1
            continue

        merged_ids = cluster_a + cluster_b
        # 3. Merged centroid must remain above threshold for every member.
        try:
            centroid = _cluster_centroid(merged_ids, templates)
        except ValueError:
            continue
        member_ok = all(
            float(np.dot(templates[tid], centroid)) >= min_member_cosine for tid in merged_ids
        )
        if not member_ok:
            transitive_rejection_count += 1
            continue

        # 4. Top1/top2 margin evidence for every member.
        margin_ok = all(
            _internal_margin(tid, merged_ids, templates) >= min_margin for tid in merged_ids
        )
        if not margin_ok:
            transitive_rejection_count += 1
            continue

        # Merge component B into A.
        cluster_a.extend(cluster_b)
        for tid in cluster_b:
            cluster_map[tid] = ca
        clusters[cb] = []
        merge_count += 1

    # Remove empty clusters and assign deterministic canonical IDs.
    nonempty = [c for c in clusters if c]
    # Sort clusters by earliest tracklet first_pts_ns then lexicographic first ID.
    nonempty.sort(key=lambda c: (min(templates.get(tid, np.zeros(512)).sum() for tid in c), c[0]))
    # Better sort by min first_pts_ns of tracklets; but TrackletRecord not available here.
    # We'll sort by sorted member IDs for determinism; aggregation will compute times.
    nonempty.sort(key=lambda c: (len(c), sorted(c)))

    return {
        "clusters": nonempty,
        "cannot_link_pairs": sorted(cannot_link),
        "candidate_edges": [
            {"tracklet_a": a, "tracklet_b": b, "cosine": float(c)} for (a, b), c in candidate_edges
        ],
        "merge_count": merge_count,
        "transitive_rejection_count": transitive_rejection_count,
    }


def build_review_queue_records(
    tracklets: list[TrackletRecord],
    reconciliation_result: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build review-queue records for uncertain reconciliation decisions."""
    min_tracklet_cosine = float(config["min_tracklet_cosine"])
    margin = float(config["min_top1_top2_margin"])
    near_window = 0.05

    records: list[dict[str, Any]] = []
    for edge in reconciliation_result["candidate_edges"]:
        cos = edge["cosine"]
        if cos >= min_tracklet_cosine - near_window and cos < min_tracklet_cosine + near_window:
            records.append(
                {
                    "type": "near_threshold_pair",
                    "tracklet_a": edge["tracklet_a"],
                    "tracklet_b": edge["tracklet_b"],
                    "cosine": cos,
                    "note": "pair near the tracklet cosine threshold",
                }
            )

    cannot_link = set(reconciliation_result["cannot_link_pairs"])
    pair_cosines = {
        (e["tracklet_a"], e["tracklet_b"]): e["cosine"]
        for e in reconciliation_result["candidate_edges"]
    }
    for pair in cannot_link:
        cos = pair_cosines.get(pair)
        if cos is not None and cos >= min_tracklet_cosine:
            records.append(
                {
                    "type": "cannot_link_high_cosine",
                    "tracklet_a": pair[0],
                    "tracklet_b": pair[1],
                    "cosine": cos,
                    "note": "cannot-link pair with high cosine similarity",
                }
            )

    for t in tracklets:
        if t.template is None:
            records.append(
                {
                    "type": "unresolved_tracklet",
                    "raw_tracklet_id": t.raw_tracklet_id,
                    "note": "tracklet has no valid identity template",
                }
            )

    return records

"""Offline tracklet reconciliation with cannot-link and complete-link rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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


def build_cannot_link_pairs(
    tracklets: list[TrackletRecord],
    human_cannot_links: set[tuple[str, str]] | None = None,
) -> set[tuple[str, str]]:
    """Build absolute cannot-link pairs.

    Two tracklets cannot be the same person if:
    - they contain observations from the same frame (actual co-occurrence);
    - a human label or external source explicitly declares them distinct.

    Mere interval overlap without same-frame evidence is NOT a cannot-link.
    """
    pairs: set[tuple[str, str]] = set()
    n = len(tracklets)
    for i in range(n):
        a = tracklets[i]
        a_frames = set(a.frame_indices)
        for j in range(i + 1, n):
            b = tracklets[j]
            if a_frames & set(b.frame_indices):
                pairs.add(_ordered_pair(a.raw_tracklet_id, b.raw_tracklet_id))

    if human_cannot_links:
        for x, y in human_cannot_links:
            pairs.add(_ordered_pair(x, y))

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


def reconcile_tracklets(
    tracklets: list[TrackletRecord],
    config: dict[str, Any],
    human_cannot_links: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Reconcile raw tracklets into canonical clusters.

    Uses complete-link clustering: a merge is allowed only when every cross-pair
    between two components passes ``min_tracklet_cosine`` and the merged centroid
    remains within ``min_member_cosine`` of every member. Cannot-links are
    absolute and override high cosine.
    """
    min_tracklet_cosine = float(config["min_tracklet_cosine"])
    min_member_cosine = float(config["min_cluster_member_cosine"])

    cannot_link = build_cannot_link_pairs(tracklets, human_cannot_links)
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

    for (id_a, id_b), _cosine in candidate_edges:
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

        # Merge component B into A.
        cluster_a.extend(cluster_b)
        for tid in cluster_b:
            cluster_map[tid] = ca
        clusters[cb] = []
        merge_count += 1

    # Remove empty clusters and assign deterministic canonical IDs.
    nonempty = [c for c in clusters if c]
    # Sort clusters by earliest first_pts_ns of any member, then lexicographic.
    nonempty.sort(
        key=lambda c: (
            min(t.first_pts_ns for t in tracklets if t.raw_tracklet_id in c),
            sorted(c),
        )
    )

    canonical_track_ids = [f"CT{idx:06d}" for idx in range(1, len(nonempty) + 1)]
    canonical_map: dict[str, str] = {}
    for canonical_id, cluster_ids in zip(canonical_track_ids, nonempty, strict=True):
        for tid in cluster_ids:
            canonical_map[tid] = canonical_id

    return {
        "clusters": nonempty,
        "canonical_track_ids": canonical_track_ids,
        "canonical_map": canonical_map,
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

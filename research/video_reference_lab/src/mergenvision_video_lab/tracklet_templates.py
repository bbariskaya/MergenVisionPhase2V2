"""Robust tracklet template aggregation from quality-selected embeddings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mergenvision_video_lab.contracts import FaceObservation


@dataclass
class TrackletTemplate:
    """Aggregated identity template for one raw tracklet."""

    raw_tracklet_id: str
    template: np.ndarray | None = None
    candidate_count: int = 0
    selected_observation_ids: list[str] = field(default_factory=list)
    rejected_observation_ids: list[str] = field(default_factory=list)
    selected_count: int = 0
    rejected_count: int = 0
    min_quality: float = 0.0
    median_quality: float = 0.0
    max_quality: float = 0.0
    intra_tracklet_cosine_min: float | None = None
    intra_tracklet_cosine_median: float | None = None
    intra_tracklet_cosine_max: float | None = None
    template_norm: float | None = None
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_tracklet_id": self.raw_tracklet_id,
            "template": (
                self.template.astype(np.float32).tolist() if self.template is not None else None
            ),
            "candidate_count": self.candidate_count,
            "selected_observation_ids": self.selected_observation_ids,
            "rejected_observation_ids": self.rejected_observation_ids,
            "selected_count": self.selected_count,
            "rejected_count": self.rejected_count,
            "min_quality": self.min_quality,
            "median_quality": self.median_quality,
            "max_quality": self.max_quality,
            "intra_tracklet_cosine_min": self.intra_tracklet_cosine_min,
            "intra_tracklet_cosine_median": self.intra_tracklet_cosine_median,
            "intra_tracklet_cosine_max": self.intra_tracklet_cosine_max,
            "template_norm": self.template_norm,
            "failure_reason": self.failure_reason,
        }


def _pairwise_cosine(embs: np.ndarray) -> np.ndarray:
    """Return NxN cosine similarity matrix for unit-normalized embeddings."""
    return np.asarray(embs @ embs.T, dtype=np.float32)


def build_tracklet_template(
    raw_tracklet_id: str,
    observations: list[FaceObservation],
    embeddings: np.ndarray,
    config: dict[str, Any],
) -> TrackletTemplate:
    """Build a robust template from a raw tracklet's observations."""
    max_selected = int(config["max_selected_samples"])
    min_selected = int(config["min_selected_samples"])
    min_separation_ns = int(config["min_temporal_separation_ns"])
    outlier_mad_scale = float(config["outlier_mad_scale"])
    outlier_floor = float(config["outlier_absolute_cosine_floor"])
    min_quality_score = float(config.get("min_quality_score", 0.0))

    rejected_ids: list[str] = []
    candidates: list[FaceObservation] = []
    for obs in observations:
        if not obs.recognition_eligible or obs.embedding_index is None:
            rejected_ids.append(obs.observation_id)
            continue
        if obs.embedding_index < 0 or obs.embedding_index >= embeddings.shape[0]:
            rejected_ids.append(obs.observation_id)
            continue
        emb = embeddings[obs.embedding_index]
        if not np.all(np.isfinite(emb)):
            rejected_ids.append(obs.observation_id)
            continue
        norm = float(np.linalg.norm(emb))
        if not np.isclose(norm, 1.0, atol=1e-4):
            rejected_ids.append(obs.observation_id)
            continue
        if obs.quality.composite_quality_score < min_quality_score:
            rejected_ids.append(obs.observation_id)
            continue
        candidates.append(obs)

    if len(candidates) < min_selected:
        return TrackletTemplate(
            raw_tracklet_id=raw_tracklet_id,
            candidate_count=len(candidates),
            rejected_observation_ids=rejected_ids,
            rejected_count=len(rejected_ids),
            failure_reason="insufficient_recognition_eligible_observations",
        )

    pts_list = [obs.pts_ns for obs in candidates]
    min_pts = min(pts_list)
    max_pts = max(pts_list)
    span_ns = max(1, max_pts - min_pts)

    # Divide span into up to max_selected bins and pick the best-quality candidate
    # from each non-empty bin.
    selected: list[FaceObservation] = []
    bin_count = max_selected
    for bin_idx in range(bin_count):
        bin_start = min_pts + (span_ns * bin_idx) // bin_count
        bin_end = min_pts + (span_ns * (bin_idx + 1)) // bin_count
        in_bin = [
            obs
            for obs in candidates
            if bin_start <= obs.pts_ns < bin_end
            or (bin_idx == bin_count - 1 and obs.pts_ns == bin_end)
        ]
        if not in_bin:
            continue
        best = max(in_bin, key=lambda o: o.quality.composite_quality_score)
        selected.append(best)

    # Fill remaining slots with highest-quality candidates while respecting
    # temporal separation.
    used_ids = {obs.observation_id for obs in selected}
    remaining = sorted(
        [obs for obs in candidates if obs.observation_id not in used_ids],
        key=lambda o: o.quality.composite_quality_score,
        reverse=True,
    )
    for obs in remaining:
        if len(selected) >= max_selected:
            break
        if all(abs(obs.pts_ns - s.pts_ns) >= min_separation_ns for s in selected):
            selected.append(obs)

    if len(selected) < min_selected:
        rejected_observation_ids = list(
            dict.fromkeys(
                rejected_ids
                + [obs.observation_id for obs in candidates if obs.observation_id not in used_ids]
            )
        )
        return TrackletTemplate(
            raw_tracklet_id=raw_tracklet_id,
            candidate_count=len(candidates),
            rejected_observation_ids=rejected_observation_ids,
            rejected_count=len(rejected_observation_ids),
            failure_reason="not_enough_temporally_separated_candidates",
        )

    selected_embs = np.asarray(
        [embeddings[obs.embedding_index] for obs in selected], dtype=np.float32
    )
    selected_ids = [obs.observation_id for obs in selected]
    qualities = np.asarray(
        [obs.quality.composite_quality_score for obs in selected], dtype=np.float64
    )

    # Medoid selection using median pairwise cosine.
    sim_matrix = _pairwise_cosine(selected_embs)
    median_sim = np.median(sim_matrix, axis=1)
    medoid_idx = int(np.argmax(median_sim))
    medoid_emb = selected_embs[medoid_idx]

    cosine_to_medoid = selected_embs @ medoid_emb
    median_cosine = float(np.median(cosine_to_medoid))
    mad = float(np.median(np.abs(cosine_to_medoid - median_cosine)))
    lower_bound = median_cosine - outlier_mad_scale * mad
    keep_threshold = max(lower_bound, outlier_floor)

    kept_mask = cosine_to_medoid >= keep_threshold
    kept_indices = np.where(kept_mask)[0].tolist()

    kept_ids = {selected_ids[i] for i in kept_indices}
    rejected_observation_ids = list(
        dict.fromkeys(
            rejected_ids
            + [obs.observation_id for obs in candidates if obs.observation_id not in kept_ids]
        )
    )

    if len(kept_indices) < min_selected:
        return TrackletTemplate(
            raw_tracklet_id=raw_tracklet_id,
            candidate_count=len(candidates),
            selected_observation_ids=[obs_id for obs_id in selected_ids if obs_id in kept_ids],
            rejected_observation_ids=rejected_observation_ids,
            selected_count=len(kept_indices),
            rejected_count=len(rejected_observation_ids),
            failure_reason="too_many_outliers_after_mad_rejection",
        )

    kept_embs = selected_embs[kept_indices]
    kept_qualities = qualities[kept_indices]
    weights = np.maximum(kept_qualities, 1e-6)
    centroid = np.average(kept_embs, axis=0, weights=weights)
    norm = float(np.linalg.norm(centroid))
    if norm <= 0.0 or not np.isfinite(norm):
        return TrackletTemplate(
            raw_tracklet_id=raw_tracklet_id,
            candidate_count=len(candidates),
            rejected_observation_ids=rejected_observation_ids,
            rejected_count=len(rejected_observation_ids),
            failure_reason="centroid_normalization_failed",
        )
    template = (centroid / norm).astype(np.float32)

    intra_min = float(np.min(cosine_to_medoid))
    intra_median = float(np.median(cosine_to_medoid))
    intra_max = float(np.max(cosine_to_medoid))

    return TrackletTemplate(
        raw_tracklet_id=raw_tracklet_id,
        template=template,
        candidate_count=len(candidates),
        selected_observation_ids=[selected_ids[i] for i in kept_indices],
        rejected_observation_ids=rejected_observation_ids,
        selected_count=len(kept_indices),
        rejected_count=len(rejected_observation_ids),
        min_quality=float(np.min(qualities)),
        median_quality=float(np.median(qualities)),
        max_quality=float(np.max(qualities)),
        intra_tracklet_cosine_min=intra_min,
        intra_tracklet_cosine_median=intra_median,
        intra_tracklet_cosine_max=intra_max,
        template_norm=float(np.linalg.norm(template)),
    )

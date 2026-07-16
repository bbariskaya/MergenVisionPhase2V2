"""Optional local gallery matching using the same oracle/alignment contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mergenvision_video_lab.oracle.insightface_oracle import InsightFaceOracle
from mergenvision_video_lab.quality import compute_quality
from mergenvision_video_lab.tracklet_templates import TrackletTemplate, build_tracklet_template


def _robust_centroid(
    embeddings: list[np.ndarray],
    qualities: list[float],
) -> np.ndarray | None:
    """Quality-weighted, L2-normalized centroid of unit embeddings."""
    if not embeddings:
        return None
    embs = np.asarray(embeddings, dtype=np.float32)
    weights = np.maximum(np.asarray(qualities, dtype=np.float64), 1e-6)
    centroid = np.average(embs, axis=0, weights=weights)
    norm = float(np.linalg.norm(centroid))
    if norm <= 0.0 or not np.isfinite(norm):
        return None
    return (centroid / norm).astype(np.float32)


def _identity_template_from_samples(
    samples: list[dict[str, Any]],
    template_config: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate multiple gallery samples into one identity template."""
    if len(samples) == 1:
        return {
            "template": samples[0]["embedding"].copy(),
            "selected": [samples[0]["sample_id"]],
            "rejected": [],
            "valid_count": 1,
        }

    # Reuse the same robust aggregation used for video tracklets by fabricating
    # synthetic FaceObservation-like objects that carry the required fields.
    from mergenvision_video_lab.contracts import (
        BBoxXYXY,
        FaceObservation,
        Landmarks5,
        QualityMetrics,
    )

    synthetic_observations: list[FaceObservation] = []
    embeddings: list[np.ndarray] = []
    for idx, sample in enumerate(samples):
        emb = sample["embedding"]
        embeddings.append(emb)
        x1, y1, x2, y2 = sample["bbox_xyxy"]
        synthetic_observations.append(
            FaceObservation(
                observation_id=sample["sample_id"],
                source_id="gallery",
                frame_index=idx,
                pts=idx,
                time_base_num=1,
                time_base_den=1,
                pts_ns=idx,
                frame_width=int(x2),
                frame_height=int(y2),
                detection_ordinal=0,
                bbox_xyxy=BBoxXYXY(x1=x1, y1=y1, x2=x2, y2=y2),
                detector_score=sample["detector_score"],
                landmarks_5=Landmarks5(
                    left_eye=tuple(sample["landmarks_5"][0]),
                    right_eye=tuple(sample["landmarks_5"][1]),
                    nose=tuple(sample["landmarks_5"][2]),
                    left_mouth=tuple(sample["landmarks_5"][3]),
                    right_mouth=tuple(sample["landmarks_5"][4]),
                ),
                quality=QualityMetrics(
                    bbox_width_px=x2 - x1,
                    bbox_height_px=y2 - y1,
                    bbox_min_side_px=min(x2 - x1, y2 - y1),
                    bbox_area_px=(x2 - x1) * (y2 - y1),
                    detector_score=sample["detector_score"],
                    grayscale_laplacian_variance=sample["quality"]["grayscale_laplacian_variance"],
                    brightness_mean=sample["quality"]["brightness_mean"],
                    brightness_std=sample["quality"]["brightness_std"],
                    dark_clip_fraction=sample["quality"]["dark_clip_fraction"],
                    bright_clip_fraction=sample["quality"]["bright_clip_fraction"],
                    interocular_distance_px=sample["quality"]["interocular_distance_px"],
                    alignment_reprojection_error_px=sample["quality"][
                        "alignment_reprojection_error_px"
                    ],
                    alignment_error_normalized_by_interocular=sample["quality"][
                        "alignment_error_normalized_by_interocular"
                    ],
                    landmark_geometry_valid=True,
                    finite_embedding=True,
                    composite_quality_score=sample["quality"]["composite_quality_score"],
                ),
                tracking_eligible=True,
                recognition_eligible=True,
                embedding_index=idx,
            )
        )

    embeddings_array = np.asarray(embeddings, dtype=np.float32)
    template = build_tracklet_template(
        raw_tracklet_id="gallery_identity",
        observations=synthetic_observations,
        embeddings=embeddings_array,
        config=template_config,
    )
    return {
        "template": template.template,
        "selected": template.selected_observation_ids,
        "rejected": template.rejected_observation_ids,
        "valid_count": template.selected_count,
    }


def build_gallery(
    oracle: InsightFaceOracle,
    gallery_root: Path | str,
    quality_config: dict[str, Any],
    template_config: dict[str, Any],
) -> dict[str, Any]:
    """Scan ``gallery_root/<identity>/<images>`` and build identity templates."""
    gallery_root = Path(gallery_root)
    result: dict[str, Any] = {
        "identities": {},
        "rejected_images": [],
        "valid_identity_count": 0,
        "strict": False,
    }

    if not gallery_root.exists():
        return result

    identity_dirs = sorted([p for p in gallery_root.iterdir() if p.is_dir()])
    if not identity_dirs:
        return result

    for identity_dir in identity_dirs:
        label = identity_dir.name
        samples: list[dict[str, Any]] = []
        for img_path in sorted(identity_dir.iterdir()):
            if not img_path.is_file():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                result["rejected_images"].append({"path": str(img_path), "reason": "unreadable"})
                continue
            h, w = img.shape[:2]
            dets = oracle.detect(
                image=img,
                detector_low_threshold=0.5,
                frame_width=w,
                frame_height=h,
                quality_config=quality_config,
                compute_embeddings=True,
            )
            if len(dets) != 1:
                result["rejected_images"].append(
                    {
                        "path": str(img_path),
                        "reason": f"detected_{len(dets)}_faces",
                    }
                )
                continue
            det = dets[0]
            if det.embedding is None:
                result["rejected_images"].append(
                    {"path": str(img_path), "reason": "no_recognition_embedding"}
                )
                continue
            samples.append(
                {
                    "sample_id": f"{label}/{img_path.name}",
                    "path": str(img_path),
                    "embedding": det.embedding.astype(np.float32),
                    "bbox_xyxy": det.bbox_xyxy,
                    "detector_score": det.detector_score,
                    "landmarks_5": det.landmarks_5.to_array(),
                    "quality": det.quality,
                }
            )

        if not samples:
            continue

        agg = _identity_template_from_samples(samples, template_config)
        if agg["template"] is None:
            result["rejected_images"].append(
                {"identity": label, "reason": "template_aggregation_failed"}
            )
            continue

        result["identities"][label] = {
            "template": agg["template"],
            "samples": [s["sample_id"] for s in samples],
            "selected_samples": agg["selected"],
            "rejected_samples": agg["rejected"],
            "valid_sample_count": len(samples),
        }
        result["valid_identity_count"] += 1

    return result


def match_cluster_to_gallery(
    cluster_template: np.ndarray | None,
    gallery: dict[str, Any],
    match_threshold: float,
    margin_threshold: float,
    min_identity_count_for_strict: int,
    min_samples_per_identity: int,
) -> dict[str, Any] | None:
    """Return gallery top1/top2 decision for a canonical cluster template."""
    if cluster_template is None:
        return None
    identities = gallery.get("identities", {})
    if not identities:
        return None

    # Strict mode requires at least N identities and min samples each.
    strict_identities = [
        label
        for label, info in identities.items()
        if info.get("valid_sample_count", 0) >= min_samples_per_identity
    ]
    strict = len(strict_identities) >= min_identity_count_for_strict and min_identity_count_for_strict > 0

    scores: list[tuple[str, float]] = []
    for label, info in identities.items():
        tpl = info["template"]
        cosine = float(np.dot(cluster_template, tpl))
        scores.append((label, cosine))
    scores.sort(key=lambda x: x[1], reverse=True)

    if not scores:
        return None

    top1_label, top1_cosine = scores[0]
    top2_label, top2_cosine = scores[1] if len(scores) > 1 else (None, 0.0)
    margin = top1_cosine - top2_cosine

    known = (
        top1_cosine >= match_threshold
        and margin >= margin_threshold
        and (not strict or (top2_label is not None and top1_label in strict_identities))
    )

    return {
        "known": known,
        "top1_label": top1_label,
        "top1_cosine": top1_cosine,
        "top2_label": top2_label,
        "top2_cosine": top2_cosine,
        "margin": margin,
        "threshold": match_threshold,
        "margin_threshold": margin_threshold,
        "strict": strict,
    }

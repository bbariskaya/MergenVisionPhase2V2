"""Final report assembly for the video reference lab."""

from __future__ import annotations

from typing import Any

from mergenvision_video_lab.contracts import RunManifest


def build_final_report(
    manifest: RunManifest,
    tracking: dict[str, Any],
    reconciliation: dict[str, Any],
    evaluation: dict[str, Any],
    benchmark: dict[str, Any],
    gallery: dict[str, Any] | None,
    rachel_proof: dict[str, Any] | None,
    verdict: str,
    limitations: list[str],
) -> dict[str, Any]:
    """Assemble the auditable final report."""
    return {
        "schema_version": "mv-video-reference-report/v1",
        "verdict": verdict,
        "manifest_summary": {
            "run_id": manifest.run_id,
            "video_sha256": manifest.video_sha256,
            "config_sha256": manifest.config_sha256,
            "logical_video_name": manifest.logical_video_name,
            "decoded_frame_count": manifest.decoded_frame_count,
            "observation_count": manifest.observation_count,
            "valid_embedding_count": manifest.valid_embedding_count,
            "providers_actual": manifest.providers_actual,
            "scene_cut_frame_count": len(manifest.scene_cut_frame_indices),
        },
        "tracking": tracking,
        "reconciliation": reconciliation,
        "evaluation": evaluation,
        "benchmark": benchmark,
        "gallery": gallery,
        "rachel_proof": rachel_proof,
        "limitations": limitations,
        "generated_at": None,  # caller may stamp if desired; excluded from deterministic hash
    }


def build_rachel_proof_table(
    early: dict[str, Any] | None,
    late: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the Rachel early/late evidence table."""
    if early is None or late is None:
        return {"status": "NOT_EVALUATED", "reason": "no labeled ground truth or gallery"}

    def row(data: dict[str, Any]) -> dict[str, Any]:
        obs = data.get("observation")
        return {
            "observation_id": obs.observation_id if obs else None,
            "frame_index": obs.frame_index if obs else None,
            "pts_ns": obs.pts_ns if obs else None,
            "raw_tracklet_id": data.get("raw_tracklet_id"),
            "canonical_track_id": data.get("canonical_track_id"),
            "gallery_top1_label": data.get("gallery_top1_label"),
            "top1_cosine": data.get("top1_cosine"),
            "top2_cosine": data.get("top2_cosine"),
            "margin": data.get("margin"),
            "template_quality": data.get("template_quality"),
        }

    return {
        "status": "EVALUATED",
        "same_raw_tracklet_required": False,
        "same_canonical_track_evidence": (
            early.get("canonical_track_id") == late.get("canonical_track_id")
        ),
        "same_persistent_face_id_proven": False,
        "early": row(early),
        "late": row(late),
    }

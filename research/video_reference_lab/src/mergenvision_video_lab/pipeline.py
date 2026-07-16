"""End-to-end reference pipeline orchestration.

This module wires extraction, replay, templates, reconciliation, gallery,
evaluation, visualization and benchmarking into discrete CLI-callable stages.
All heavy imports happen inside stage functions so that the module can be
imported for --help without loading ML frameworks.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from mergenvision_video_lab.aggregation import aggregate_canonical_tracks
from mergenvision_video_lab.artifact_store import ArtifactStore
from mergenvision_video_lab.config import LabConfig, resolve_repo_relative_path
from mergenvision_video_lab.contracts import CanonicalTrack, FaceObservation
from mergenvision_video_lab.errors import ErrorCode, LabError
from mergenvision_video_lab.extraction import extract_video, resolve_run_dir
from mergenvision_video_lab.gallery import build_gallery, match_cluster_to_gallery
from mergenvision_video_lab.reconciliation import TrackletRecord, reconcile_tracklets
from mergenvision_video_lab.replay import replay_frames
from mergenvision_video_lab.tracklet_templates import build_tracklet_template
from mergenvision_video_lab.visualization import (
    make_alignment_contact_sheet,
    make_canonical_contact_sheet,
    make_cosine_histograms,
    make_overlay_jsonl,
    make_quality_histograms,
    make_timeline,
    make_tracklet_contact_sheet,
    render_debug_mp4,
)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _observations_by_tracklet(
    observations: list[FaceObservation],
    assignments: list[dict[str, Any]],
) -> dict[str, list[FaceObservation]]:
    obs_by_id = {obs.observation_id: obs for obs in observations}
    grouped: dict[str, list[FaceObservation]] = {}
    for assignment in assignments:
        obs = obs_by_id.get(assignment["observation_id"])
        if obs is None:
            continue
        grouped.setdefault(assignment["raw_tracklet_id"], []).append(obs)
    return grouped


def _tracklets_to_records(
    tracklets: list[Any],
    templates: dict[str, np.ndarray | None],
) -> list[TrackletRecord]:
    records: list[TrackletRecord] = []
    for t in tracklets:
        records.append(
            TrackletRecord(
                raw_tracklet_id=t.raw_tracklet_id,
                strategy=t.strategy,
                observation_ids=t.observation_ids,
                frame_indices=t.frame_indices,
                first_pts_ns=t.start_pts_ns or 0,
                last_pts_ns=t.last_pts_ns or 0,
                template=templates.get(t.raw_tracklet_id),
            )
        )
    return records


def stage_extract(config: LabConfig, force: bool = False) -> dict[str, Any]:
    """Run reference extraction or resume from valid artifacts."""
    store, manifest = extract_video(config, force=force)
    return {
        "run_dir": str(store.run_dir),
        "manifest": manifest.model_dump(mode="json"),
    }


def stage_validate(config: LabConfig) -> dict[str, Any]:
    """Validate frozen artifacts for the configured video."""
    run_dir = resolve_run_dir(config)
    store = ArtifactStore(run_dir)
    store.validate()
    manifest = store.read_manifest()
    return {
        "run_dir": str(run_dir),
        "valid": True,
        "manifest": manifest.model_dump(mode="json"),
    }


def stage_replay(
    config: LabConfig, strategy: str = "byte_iou", chunk_size: int = 1
) -> dict[str, Any]:
    """Replay frozen observations through a tracker."""
    run_dir = resolve_run_dir(config)
    result = replay_frames(
        run_dir,
        config.tracking.model_dump(mode="json"),
        strategy=strategy,
        chunk_size=chunk_size,
    )
    # Write replay artifacts.
    replay_dir = Path(run_dir) / "replay" / strategy
    _ensure_dir(replay_dir)
    with open(replay_dir / "assignments.jsonl", "w", encoding="utf-8") as f:
        for assignment in result["assignments"]:
            f.write(json.dumps(assignment) + "\n")
    with open(replay_dir / "tracklets.jsonl", "w", encoding="utf-8") as f:
        for tracklet in result["tracklets"]:
            f.write(json.dumps(tracklet) + "\n")
    return {**result, "replay_dir": str(replay_dir)}


def stage_templates(config: LabConfig, strategy: str = "byte_iou") -> dict[str, Any]:
    """Build tracklet templates from frozen observations and replay assignments."""
    run_dir = resolve_run_dir(config)
    store = ArtifactStore(run_dir)
    store.validate()

    observations = store.read_observations()
    embeddings = store.read_embeddings()

    replay_dir = run_dir / "replay" / strategy
    if not (replay_dir / "assignments.jsonl").exists():
        stage_replay(config, strategy=strategy, chunk_size=1)

    assignments: list[dict[str, Any]] = []
    with open(replay_dir / "assignments.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                assignments.append(json.loads(line))

    grouped = _observations_by_tracklet(observations, assignments)
    template_config = config.templates.model_dump(mode="json")

    templates: dict[str, np.ndarray | None] = {}
    template_records: list[dict[str, Any]] = []
    failed_tracklets: list[str] = []
    for tid, obs_list in grouped.items():
        template = build_tracklet_template(tid, obs_list, embeddings, template_config)
        templates[tid] = template.template
        if template.template is None:
            failed_tracklets.append(tid)
        template_records.append(template.to_dict())

    tracks_dir = run_dir / "tracks"
    _ensure_dir(tracks_dir)
    with open(tracks_dir / "tracklet_templates.jsonl", "w", encoding="utf-8") as f:
        for record in template_records:
            f.write(json.dumps(record, default=_json_default) + "\n")

    valid_templates = [t for t in templates.values() if t is not None]
    return {
        "strategy": strategy,
        "tracklet_count": len(grouped),
        "valid_template_count": len(valid_templates),
        "failed_tracklet_count": len(failed_tracklets),
        "tracks_dir": str(tracks_dir),
    }


def stage_reconcile(config: LabConfig, strategy: str = "byte_iou") -> dict[str, Any]:
    """Reconcile raw tracklets into canonical tracks."""
    from mergenvision_video_lab.tracking.byte_tracker import ByteTrackIoUTracker

    run_dir = resolve_run_dir(config)
    store = ArtifactStore(run_dir)
    store.validate()

    observations = store.read_observations()
    embeddings = store.read_embeddings()
    frames = store.read_frames()

    replay_dir = run_dir / "replay" / strategy
    if not (replay_dir / "assignments.jsonl").exists():
        stage_replay(config, strategy=strategy, chunk_size=1)

    assignments: list[dict[str, Any]] = []
    with open(replay_dir / "assignments.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                assignments.append(json.loads(line))

    tracks_dir = run_dir / "tracks"
    if not (tracks_dir / "tracklet_templates.jsonl").exists():
        stage_templates(config, strategy=strategy)

    # Re-run replay to obtain live Tracklet objects for summaries.
    tracker = ByteTrackIoUTracker(config.tracking.model_dump(mode="json"))
    grouped: dict[int, list[FaceObservation]] = {}
    for obs in observations:
        grouped.setdefault(obs.frame_index, []).append(obs)
    for frame in frames:
        tracker.update(
            frame.frame_index,
            frame.pts_ns,
            grouped.get(frame.frame_index, []),
            embeddings,
            scene_cut_before=frame.scene_cut_before,
        )
    tracker.finalize()
    tracklets = tracker.removed_tracklets()

    template_records: list[dict[str, Any]] = []
    with open(tracks_dir / "tracklet_templates.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                template_records.append(json.loads(line))

    templates_by_id: dict[str, np.ndarray | None] = {}
    for record in template_records:
        tpl = record.get("template")
        if tpl is not None:
            templates_by_id[record["raw_tracklet_id"]] = np.asarray(tpl, dtype=np.float32)
        else:
            templates_by_id[record["raw_tracklet_id"]] = None

    records = _tracklets_to_records(tracklets, templates_by_id)
    result = reconcile_tracklets(
        records,
        config.reconciliation.model_dump(mode="json"),
    )

    with open(tracks_dir / "reconciliation_pairs.jsonl", "w", encoding="utf-8") as f:
        for edge in result["candidate_edges"]:
            f.write(json.dumps(edge) + "\n")

    with open(tracks_dir / "canonical_tracks.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "canonical_track_ids": result["canonical_track_ids"],
                "clusters": result["clusters"],
                "canonical_map": result["canonical_map"],
                "cannot_link_pairs": result["cannot_link_pairs"],
                "merge_count": result["merge_count"],
                "transitive_rejection_count": result["transitive_rejection_count"],
            },
            f,
            indent=2,
        )

    canonical_tracks = aggregate_canonical_tracks(
        observations,
        assignments,
        result["clusters"],
        result["canonical_track_ids"],
        templates_by_id,
        config.appearances.max_gap_multiplier,
    )
    with open(tracks_dir / "canonical_tracks.jsonl", "w", encoding="utf-8") as f:
        for track in canonical_tracks:
            f.write(track.model_dump_json() + "\n")

    return {
        "strategy": strategy,
        "cluster_count": len(result["clusters"]),
        "merge_count": result["merge_count"],
        "transitive_rejection_count": result["transitive_rejection_count"],
        "cannot_link_count": len(result["cannot_link_pairs"]),
        "tracks_dir": str(tracks_dir),
    }


def stage_gallery(config: LabConfig, strategy: str = "byte_iou") -> dict[str, Any] | None:
    """Build gallery templates and annotate canonical tracks.

    Returns None if no gallery root exists or no valid identities are found.
    """
    from mergenvision_video_lab.oracle.insightface_oracle import InsightFaceOracle

    run_dir = resolve_run_dir(config)
    store = ArtifactStore(run_dir)
    store.validate()

    gallery_root = Path(resolve_repo_relative_path(config.gallery.root))
    if not gallery_root.exists():
        return None

    oracle = InsightFaceOracle(
        model_root=config.oracle.model_root,
        model_pack=config.oracle.model_pack,
        requested_provider=config.oracle.provider,
        allow_cpu_fallback=config.oracle.allow_cpu_fallback,
        det_size=tuple(config.oracle.det_size),
    )

    gallery = build_gallery(
        oracle,
        gallery_root,
        config.quality.model_dump(mode="json"),
        config.templates.model_dump(mode="json"),
    )
    if gallery.get("valid_identity_count", 0) == 0:
        return None

    tracks_path = run_dir / "tracks" / "canonical_tracks.json"
    if not tracks_path.exists():
        stage_reconcile(config, strategy=strategy)

    with open(tracks_path, encoding="utf-8") as f:
        reconciliation_data = json.load(f)

    cluster_ids = reconciliation_data["canonical_track_ids"]
    clusters = reconciliation_data["clusters"]
    canonical_map: dict[str, str] = reconciliation_data["canonical_map"]

    template_records: list[dict[str, Any]] = []
    with open(run_dir / "tracks" / "tracklet_templates.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                template_records.append(json.loads(line))

    templates_by_id: dict[str, np.ndarray | None] = {}
    for record in template_records:
        tpl = record.get("template")
        if tpl is not None:
            templates_by_id[record["raw_tracklet_id"]] = np.asarray(tpl, dtype=np.float32)

    canonical_templates: dict[str, np.ndarray] = {}
    for canonical_id, member_ids in zip(cluster_ids, clusters, strict=True):
        member_temps = [
            templates_by_id[tid] for tid in member_ids if templates_by_id.get(tid) is not None
        ]
        if not member_temps:
            continue
        centroid = np.mean(np.asarray(member_temps, dtype=np.float32), axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            canonical_templates[canonical_id] = (centroid / norm).astype(np.float32)

    decisions: dict[str, dict[str, Any]] = {}
    for canonical_id, template in canonical_templates.items():
        match = match_cluster_to_gallery(
            template,
            gallery,
            config.gallery.match_threshold,
            config.gallery.match_margin,
            config.gallery.min_identity_count_for_strict_margin,
            config.gallery.min_samples_per_identity,
        )
        if match is not None:
            decisions[canonical_id] = match

    labels = {
        cid: (match["top1_label"] if match.get("known") else None)
        for cid, match in decisions.items()
    }

    # Apply gallery decisions to canonical tracks and rewrite the ledger.
    canonical_tracks_path = run_dir / "tracks" / "canonical_tracks.jsonl"
    updated_tracks: list[CanonicalTrack] = []
    if canonical_tracks_path.exists():
        with open(canonical_tracks_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    updated_tracks.append(CanonicalTrack.model_validate_json(line))
        for track in updated_tracks:
            match = decisions.get(track.canonical_track_id)
            if match is None:
                continue
            track.display_label = labels.get(track.canonical_track_id)
            track.gallery_top1_label = match.get("top1_label")
            track.gallery_top1_cosine = match.get("top1_cosine")
            track.gallery_top2_label = match.get("top2_label")
            track.gallery_top2_cosine = match.get("top2_cosine")
            track.gallery_margin = match.get("margin")
            track.decision_reason = "gallery_match" if match.get("known") else "gallery_rejected"
            track.confidence_evidence["match_threshold"] = match.get("threshold")
            track.confidence_evidence["margin_threshold"] = match.get("margin_threshold")
            track.confidence_evidence["strict"] = match.get("strict")
        with open(canonical_tracks_path, "w", encoding="utf-8") as f:
            for track in updated_tracks:
                f.write(track.model_dump_json() + "\n")

    gallery_dir = run_dir / "gallery"
    _ensure_dir(gallery_dir)
    with open(gallery_dir / "decisions.json", "w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2)
    with open(gallery_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    with open(gallery_dir / "canonical_map.json", "w", encoding="utf-8") as f:
        json.dump(canonical_map, f, indent=2)

    return {
        "strategy": strategy,
        "identity_count": gallery["valid_identity_count"],
        "decision_count": len(decisions),
        "known_count": sum(1 for d in decisions.values() if d.get("known")),
        "gallery_dir": str(gallery_dir),
    }


def stage_evaluate(config: LabConfig, strategy: str = "byte_iou") -> dict[str, Any]:
    """Evaluate tracking and identity metrics from frozen artifacts."""
    from mergenvision_video_lab.evaluation import (
        evaluate_gallery,
        evaluate_identity,
        evaluate_tracking,
    )

    run_dir = resolve_run_dir(config)
    store = ArtifactStore(run_dir)
    store.validate()

    observations = store.read_observations()
    replay_dir = run_dir / "replay" / strategy
    assignments: list[dict[str, Any]] = []
    with open(replay_dir / "assignments.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                assignments.append(json.loads(line))

    tracking_metrics = evaluate_tracking(observations, assignments)

    tracks_path = run_dir / "tracks" / "canonical_tracks.json"
    canonical_tracks: list[CanonicalTrack] = []
    clusters: list[list[str]] = []
    if tracks_path.exists():
        with open(tracks_path, encoding="utf-8") as f:
            reconciliation_data = json.load(f)
        clusters = reconciliation_data["clusters"]
        canonical_tracks_path = run_dir / "tracks" / "canonical_tracks.jsonl"
        if canonical_tracks_path.exists():
            with open(canonical_tracks_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        canonical_tracks.append(CanonicalTrack.model_validate_json(line))

    identity_metrics = evaluate_identity(
        clusters,
        assignments=assignments,
        canonical_map=reconciliation_data.get("canonical_map", {}),
        ground_truth=None,
        resolved_anchors=None,
    )
    gallery_metrics = evaluate_gallery(canonical_tracks)

    eval_dir = run_dir / "evaluation"
    _ensure_dir(eval_dir)
    metrics = {
        "tracking": tracking_metrics,
        "identity": identity_metrics,
        "gallery": gallery_metrics,
    }
    with open(eval_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {"strategy": strategy, "metrics": metrics, "eval_dir": str(eval_dir)}


def stage_visualize(config: LabConfig, strategy: str = "byte_iou") -> dict[str, Any]:
    """Generate contact sheets, histograms, timeline, overlay and debug MP4."""
    run_dir = resolve_run_dir(config)
    store = ArtifactStore(run_dir)
    store.validate()

    video_path = resolve_repo_relative_path(config.video.path)
    observations = store.read_observations()
    embeddings = store.read_embeddings()

    replay_dir = run_dir / "replay" / strategy
    assignments: list[dict[str, Any]] = []
    with open(replay_dir / "assignments.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                assignments.append(json.loads(line))

    tracklet_summaries: list[dict[str, Any]] = []
    tracklets_path = replay_dir / "tracklets.jsonl"
    if tracklets_path.exists():
        with open(tracklets_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tracklet_summaries.append(json.loads(line))

    tracks_path = run_dir / "tracks" / "canonical_tracks.json"
    canonical_map: dict[str, str] = {}
    labels: dict[str, str | None] = {}
    canonical_tracks: list[CanonicalTrack] = []
    if tracks_path.exists():
        with open(tracks_path, encoding="utf-8") as f:
            reconciliation_data = json.load(f)
        canonical_map = reconciliation_data.get("canonical_map", {})
        labels_path = run_dir / "gallery" / "labels.json"
        if labels_path.exists():
            with open(labels_path, encoding="utf-8") as f:
                labels = json.load(f)
        canonical_tracks_path = run_dir / "tracks" / "canonical_tracks.jsonl"
        if canonical_tracks_path.exists():
            with open(canonical_tracks_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        canonical_tracks.append(CanonicalTrack.model_validate_json(line))

    visual_dir = run_dir / "visual"
    _ensure_dir(visual_dir)

    make_alignment_contact_sheet(
        video_path,
        observations,
        visual_dir / "alignment_contact_sheet.jpg",
        alignment_config=config.alignment.model_dump(mode="json"),
    )

    make_quality_histograms(observations, visual_dir / "quality_histograms.jpg")

    same_person_cosines: list[float] = []
    diff_person_cosines: list[float] = []
    # Lightweight diagnostic: compare embeddings of consecutive observations.
    for i in range(0, len(observations) - 1, 2):
        a = observations[i]
        b = observations[i + 1]
        if a.embedding_index is None or b.embedding_index is None:
            continue
        cosine = float(np.dot(embeddings[a.embedding_index], embeddings[b.embedding_index]))
        if a.frame_index == b.frame_index:
            diff_person_cosines.append(cosine)
        else:
            same_person_cosines.append(cosine)
    make_cosine_histograms(
        {
            "same_person_consecutive": same_person_cosines,
            "diff_person_same_frame": diff_person_cosines,
        },
        visual_dir / "cosine_histograms.jpg",
    )

    make_overlay_jsonl(
        observations,
        assignments,
        canonical_map,
        labels,
        visual_dir / "overlay.jsonl",
    )

    if config.render.debug_mp4:
        render_debug_mp4(
            video_path,
            observations,
            assignments,
            canonical_map,
            labels,
            visual_dir / "annotated_debug.mp4",
            preserve_audio=config.render.preserve_audio_if_possible,
        )

    make_timeline(tracklet_summaries, canonical_map, visual_dir / "timeline.jpg")
    make_tracklet_contact_sheet(
        video_path,
        tracklet_summaries,
        observations,
        visual_dir / "tracklet_contact_sheet.jpg",
    )
    make_canonical_contact_sheet(
        video_path,
        canonical_tracks,
        observations,
        visual_dir / "canonical_contact_sheet.jpg",
    )

    return {"strategy": strategy, "visual_dir": str(visual_dir)}


def stage_benchmark(config: LabConfig, strategy: str = "byte_iou") -> dict[str, Any]:
    """Benchmark extraction and frozen replay."""
    from mergenvision_video_lab.benchmark import benchmark_extraction, benchmark_replay
    from mergenvision_video_lab.tracking.byte_tracker import ByteTrackIoUTracker

    run_dir = resolve_run_dir(config)
    store = ArtifactStore(run_dir)
    store.validate()

    manifest = store.read_manifest()
    extraction_bench = benchmark_extraction(manifest)

    observations = store.read_observations()
    embeddings = store.read_embeddings()

    replay_benches: list[dict[str, Any]] = []
    for chunk_size in config.replay.chunk_sizes:
        bench = benchmark_replay(
            observations,
            embeddings,
            manifest,
            ByteTrackIoUTracker,
            config.tracking.model_dump(mode="json"),
            chunk_size=chunk_size,
            warmup_runs=config.benchmark.warmup_runs,
            measured_runs=config.benchmark.measured_runs,
            target_fps=config.benchmark.informational_target_fps,
        )
        replay_benches.append(bench)

    benchmark_dir = run_dir / "benchmark"
    _ensure_dir(benchmark_dir)
    result = {
        "extraction": extraction_bench,
        "replay": replay_benches,
    }
    with open(benchmark_dir / "benchmark.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return {"strategy": strategy, "benchmark": result, "benchmark_dir": str(benchmark_dir)}


def run_friends(
    config: LabConfig,
    max_frames: int | None = None,
    stages: list[str] | None = None,
) -> dict[str, Any]:
    """Run the full Friends reference pipeline end-to-end."""
    if max_frames is not None:
        config.video.max_frames = max_frames

    available_stages: dict[str, Callable[[], dict[str, Any] | None]] = {
        "extract": lambda: stage_extract(config),
        "validate": lambda: stage_validate(config),
        "replay": lambda: stage_replay(config),
        "templates": lambda: stage_templates(config),
        "reconcile": lambda: stage_reconcile(config),
        "gallery": lambda: stage_gallery(config),
        "evaluate": lambda: stage_evaluate(config),
        "visualize": lambda: stage_visualize(config),
        "benchmark": lambda: stage_benchmark(config),
    }

    selected = stages if stages is not None else list(available_stages.keys())
    results: dict[str, Any] = {}
    for stage_name in selected:
        if stage_name not in available_stages:
            raise LabError(
                ErrorCode.CONFIG_INVALID,
                f"unknown stage: {stage_name}",
            )
        t0 = time.perf_counter()
        stage_result = available_stages[stage_name]()
        if stage_result is None:
            stage_result = {}
        stage_result["elapsed_seconds"] = time.perf_counter() - t0
        results[stage_name] = stage_result

    return results


def _json_default(obj: object) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

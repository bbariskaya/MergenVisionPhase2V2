"""Reference extraction: decode -> detect -> align -> quality -> embed -> freeze."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from importlib.metadata import version as pkg_version

from mergenvision_video_lab.artifact_store import ArtifactStore, RunLock
from mergenvision_video_lab.config import LabConfig, config_sha256
from mergenvision_video_lab.contracts import (
    AlignmentContract,
    BBoxXYXY,
    ExtractionTiming,
    FaceObservation,
    Landmarks5,
    OnnxModelContract,
    QualityMetrics,
    RunManifest,
    SamplingContract,
)
from mergenvision_video_lab.errors import ArtifactCorruptError, LabError, VideoReadError
from mergenvision_video_lab.hashing import sha256_file
from mergenvision_video_lab.oracle.insightface_oracle import InsightFaceOracle
from mergenvision_video_lab.quality import compute_quality
from mergenvision_video_lab.video_reader import VideoReader


def _package_versions() -> dict[str, str]:
    names = [
        "numpy",
        "scipy",
        "av",
        "opencv-python-headless",
        "scikit-image",
        "onnx",
        "onnxruntime",
        "insightface",
        "pydantic",
        "pydantic-settings",
        "PyYAML",
        "typer",
        "orjson",
        "pandas",
        "matplotlib",
        "Pillow",
        "psutil",
        "rich",
    ]
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = pkg_version(name)
        except Exception:
            result[name] = "unknown"
    return result


def _compute_scene_score(
    prev_hist: np.ndarray | None,
    gray_small: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Bhattacharyya histogram distance between consecutive downscaled frames."""
    hist = cv2.calcHist([gray_small], [0], None, [16], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    if prev_hist is None:
        return 0.0, hist
    score = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
    return score, hist


def _make_observation(
    frame,
    det,
    detection_ordinal: int,
    scene_cut_before: bool,
    scene_change_score: float,
    quality_config: dict[str, Any],
    embeddings: list[np.ndarray],
) -> FaceObservation:
    """Build one FaceObservation from an oracle detection."""
    bbox = det.bbox_xyxy
    x1, y1, x2, y2 = bbox
    frame_w = frame.display_width
    frame_h = frame.display_height
    bbox_min_side = min(x2 - x1, y2 - y1)

    quality = compute_quality(
        aligned_crop=det.aligned_crop,
        bbox_xyxy=bbox,
        frame_width=frame_w,
        frame_height=frame_h,
        landmarks_5=det.landmarks_5.to_array(),
        detector_score=det.detector_score,
        reprojection_error_px=det.quality["alignment_reprojection_error_px"],
        config=quality_config,
    )
    quality.landmark_geometry_valid = True

    tracking_eligible = (
        bbox_min_side >= quality_config["min_face_side_px"]
        and np.isfinite(det.detector_score)
        and quality.landmark_geometry_valid
    )

    embedding: np.ndarray | None = det.embedding
    embedding_index: int | None = None
    recognition_eligible = (
        tracking_eligible
        and embedding is not None
        and len(quality.hard_rejection_reasons) == 0
    )
    if recognition_eligible:
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(norm) or not np.isclose(norm, 1.0, atol=1e-4):
            recognition_eligible = False
            quality.finite_embedding = False
            quality.hard_rejection_reasons.append("embedding_not_unit_normalized")
        else:
            embedding_index = len(embeddings)
            embeddings.append(embedding.astype(np.float32))
            quality.finite_embedding = True
    else:
        quality.finite_embedding = False
        if embedding is None:
            quality.hard_rejection_reasons.append("no_embedding")

    rejection_reasons = quality.hard_rejection_reasons.copy() if not recognition_eligible else []

    return FaceObservation(
        observation_id=f"{frame.source_id}:{frame.frame_index:08d}:{detection_ordinal:04d}",
        source_id=frame.source_id,
        frame_index=frame.frame_index,
        pts=frame.pts,
        time_base_num=frame.time_base_num,
        time_base_den=frame.time_base_den,
        pts_ns=frame.pts_ns,
        frame_width=frame_w,
        frame_height=frame_h,
        rotation_applied=frame.rotation_degrees,
        detection_ordinal=detection_ordinal,
        bbox_xyxy=BBoxXYXY(x1=x1, y1=y1, x2=x2, y2=y2),
        detector_score=det.detector_score,
        landmarks_5=det.landmarks_5,
        quality=quality,
        tracking_eligible=tracking_eligible,
        recognition_eligible=recognition_eligible,
        rejection_reasons=rejection_reasons,
        embedding_index=embedding_index,
        scene_cut_before=scene_cut_before,
    )


def _build_manifest(
    config: LabConfig,
    video_path: Path,
    video_sha256: str,
    probe,
    observations: list[FaceObservation],
    embeddings: np.ndarray,
    detector_contract: OnnxModelContract,
    recognizer_contract: OnnxModelContract,
    provider_requested: str,
    providers_available: list[str],
    providers_actual: list[str],
    extraction_timing: ExtractionTiming,
) -> RunManifest:
    rejection_counts: dict[str, int] = {}
    for obs in observations:
        if not obs.recognition_eligible:
            for reason in obs.rejection_reasons:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    sampling = SamplingContract(
        mode=config.video.sampling_mode,
        max_frames=config.video.max_frames,
    )
    if config.video.sampling_mode == "every_n_frames":
        sampling.every_n_frames = getattr(config.video, "every_n_frames", None)
    if config.video.sampling_mode == "frames_per_second":
        sampling.frames_per_second = getattr(config.video, "frames_per_second", None)

    alignment = AlignmentContract(
        output_size=config.alignment.output_size,
        color_order=config.alignment.color_order,
        border_mode=config.alignment.border_mode,
        interpolation=config.alignment.interpolation,
        landmark_order=config.alignment.landmark_order,
        arcface_template=[
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
    )

    return RunManifest(
        run_id=f"{video_sha256[:12]}_{config_sha256(config)[:12]}",
        video_sha256=video_sha256,
        video_size_bytes=video_path.stat().st_size,
        logical_video_name=video_path.name,
        container=probe.container,
        codec=probe.codec,
        pixel_format=probe.pixel_format,
        display_width=probe.display_width,
        display_height=probe.display_height,
        rotation_degrees=probe.rotation_degrees,
        stream_index=probe.video_stream_index,
        time_base_num=probe.time_base_num,
        time_base_den=probe.time_base_den,
        duration_ns=probe.duration_ns or -1,
        decoded_frame_count=0,  # filled later
        processed_frame_count=0,
        sampling_contract=sampling,
        detector_contract=detector_contract,
        recognizer_contract=recognizer_contract,
        alignment_contract=alignment,
        provider_requested=provider_requested,
        providers_available=providers_available,
        providers_actual=providers_actual,
        package_versions=_package_versions(),
        config_sha256=config_sha256(config),
        observation_count=len(observations),
        valid_embedding_count=embeddings.shape[0],
        rejection_counts=rejection_counts,
        extraction_timing=extraction_timing,
        warnings=[],
        limitations=[
            "reference_oracle only; not production GPU path",
            "scene-cut threshold is exploratory",
        ],
    )


def extract_video(
    config: LabConfig,
    force: bool = False,
) -> tuple[ArtifactStore, RunManifest]:
    """Run the full reference extraction and return the artifact store + manifest."""
    video_path = Path(config.video.path)
    if not video_path.exists():
        raise VideoReadError(f"input video not found: {video_path}")

    video_sha256 = sha256_file(video_path)
    cfg_sha = config_sha256(config)
    run_dir = (
        Path("artifacts")
        / "video_reference"
        / video_sha256[:12]
        / cfg_sha[:12]
    )
    store = ArtifactStore(run_dir)

    with RunLock(run_dir):
        if not force and store.is_complete():
            try:
                store.validate()
                manifest = store.read_manifest()
                if (
                    manifest.video_sha256 == video_sha256
                    and manifest.config_sha256 == cfg_sha
                ):
                    return store, manifest
            except ArtifactCorruptError:
                pass

        reader = VideoReader(video_path, source_id=video_path.name)
        probe = reader.probe

        oracle = InsightFaceOracle(
            model_root=config.oracle.model_root,
            model_pack=config.oracle.model_pack,
            requested_provider=config.oracle.provider,
            allow_cpu_fallback=config.oracle.allow_cpu_fallback,
            det_size=tuple(config.oracle.det_size),
        )

        quality_config = config.quality.model_dump(mode="json")
        scene_thresh = float(getattr(config.video, "scene_cut_threshold", 0.45))
        downscale = int(getattr(config.video, "scene_cut_downscale", 64))

        observations: list[FaceObservation] = []
        embeddings: list[np.ndarray] = []

        decode_seconds = 0.0
        oracle_seconds = 0.0
        quality_seconds = 0.0
        t_start = time.perf_counter()

        prev_hist: np.ndarray | None = None
        scene_cut_before = False
        scene_cut_frame_indices: list[int] = []
        decoded_frame_count = 0
        processed_frame_count = 0

        try:
            for frame in reader:
                decoded_frame_count += 1
                if config.video.max_frames is not None and decoded_frame_count > config.video.max_frames:
                    break

                t_decode = time.perf_counter()
                gray_small = cv2.cvtColor(frame.ndarray, cv2.COLOR_BGR2GRAY)
                gray_small = cv2.resize(gray_small, (downscale, downscale))
                scene_score, prev_hist = _compute_scene_score(prev_hist, gray_small)
                scene_cut_before = decoded_frame_count > 1 and scene_score > scene_thresh

                t_oracle_start = time.perf_counter()
                decode_seconds += t_oracle_start - t_decode

                dets = oracle.detect(
                    image=frame.ndarray,
                    detector_low_threshold=config.oracle.detector_low_threshold,
                    frame_width=frame.display_width,
                    frame_height=frame.display_height,
                    quality_config=quality_config,
                    compute_embeddings=True,
                )
                t_quality_start = time.perf_counter()
                oracle_seconds += t_quality_start - t_oracle_start

                # Deterministic ordering is already guaranteed by the oracle, but
                # re-sort defensively by the documented contract.
                dets.sort(
                    key=lambda d: (
                        d.bbox_xyxy[0],
                        d.bbox_xyxy[1],
                        d.bbox_xyxy[2],
                        -d.detector_score,
                    )
                )

                for ordinal, det in enumerate(dets, start=1):
                    obs = _make_observation(
                        frame,
                        det,
                        ordinal,
                        scene_cut_before,
                        scene_score,
                        quality_config,
                        embeddings,
                    )
                    observations.append(obs)
                    processed_frame_count += 1

                quality_seconds += time.perf_counter() - t_quality_start
        except VideoReadError:
            raise
        except Exception as exc:
            raise VideoReadError(f"extraction failed at frame {decoded_frame_count}: {exc}") from exc

        embeddings_array = np.asarray(embeddings, dtype=np.float32)
        if embeddings_array.ndim != 2 or embeddings_array.shape[1] != 512:
            if embeddings_array.size == 0:
                embeddings_array = np.zeros((0, 512), dtype=np.float32)
            else:
                raise LabError(
                    "unexpected embedding shape",
                    {"shape": embeddings_array.shape},
                )

        t_serialization_start = time.perf_counter()
        extraction_timing = ExtractionTiming(
            decode_seconds=decode_seconds,
            oracle_seconds=oracle_seconds,
            quality_alignment_seconds=quality_seconds,
            serialization_seconds=0.0,
            total_seconds=t_serialization_start - t_start,
        )

        manifest = _build_manifest(
            config=config,
            video_path=video_path,
            video_sha256=video_sha256,
            probe=probe,
            observations=observations,
            embeddings=embeddings_array,
            detector_contract=oracle.detector_contract,
            recognizer_contract=oracle.recognizer_contract,
            provider_requested=config.oracle.provider,
            providers_available=oracle._available_providers,
            providers_actual=oracle._actual_providers,
            extraction_timing=extraction_timing,
        )
        manifest.decoded_frame_count = decoded_frame_count
        manifest.processed_frame_count = processed_frame_count

        store.write_manifest(manifest)
        store.write_observations(observations)
        store.write_embeddings(embeddings_array)
        store.write_checksums()

        extraction_timing.serialization_seconds = time.perf_counter() - t_serialization_start
        # Re-write manifest with updated serialization timing.
        manifest.extraction_timing = extraction_timing
        store.write_manifest(manifest)

    return store, manifest

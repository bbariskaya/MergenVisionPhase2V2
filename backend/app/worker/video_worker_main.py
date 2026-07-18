from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import zstandard

from app.application.ports.video_observations import VideoObservationFrame
from app.domain.entities.video_job import VideoJob
from app.domain.value_objects import JobId
from app.infrastructure.config import settings
from app.infrastructure.persistence.sqlalchemy.session import async_session_maker
from app.infrastructure.persistence.sqlalchemy.unit_of_work import SqlAlchemyUnitOfWork
from app.infrastructure.runtime.native_representative_crop_provider import (
    NativeRepresentativeCropProvider,
)
from app.infrastructure.serialization.native_bundle_reader import (
    NativeBundle,
    NativeBundleError,
)
from app.infrastructure.serialization.video_observation_reader import (
    ObservationArtifactError,
    read_observation_frames,
)
from app.infrastructure.storage.minio_adapter import MinIOObjectStore
from app.infrastructure.uuid7 import generate_uuid7
from app.worker.bootstrap import build_video_processing_service

logger = logging.getLogger(__name__)

WORKER_ID = os.environ.get("WORKER_ID", "worker-unknown")
GPU_DEVICE_ID = int(os.environ.get("GPU_DEVICE_ID", "0"))
LEASE_SECONDS = int(os.environ.get("WORKER_LEASE_SECONDS", "1800"))
WORKER_BIN = os.environ.get(
    "MV_VIDEO_WORKER_PATH",
    "/workspace/backend/native/video_worker/build/mv_video_worker",
)


def _uow_factory() -> SqlAlchemyUnitOfWork:
    return SqlAlchemyUnitOfWork(async_session_maker)


async def _download_source_video(
    object_store: MinIOObjectStore,
    video_id: uuid.UUID,
    dest_dir: Path,
) -> Path:
    key = f"videos/{video_id}/source/original"
    dest = dest_dir / "source.mp4"
    data = await object_store.get(key)
    if data is None:
        raise ObservationArtifactError(f"source video not found in MinIO: {key}")
    dest.write_bytes(data)
    return dest


async def _run_native_worker(
    input_path: Path,
    output_dir: Path,
    job: VideoJob,
) -> None:
    job_id = job.job_id
    video_id = job.video_id
    max_frames_env = os.environ.get("MV_VIDEO_WORKER_MAX_FRAMES", "").strip()
    extra_args: list[str] = []
    if max_frames_env.isdigit():
        extra_args.extend(["--max-frames", max_frames_env])
    else:
        extra_args.append("--all-frames")

    detector_batch = os.environ.get("DETECTOR_BATCH_SIZE", "16")
    recognizer_batch = os.environ.get("RECOGNIZER_BATCH_SIZE", "32")
    cmd = [
        WORKER_BIN,
        "--input", str(input_path),
        "--output", str(output_dir),
        "--gpu-id", str(GPU_DEVICE_ID),
        "--job-id", str(job_id),
        "--video-id", str(video_id),
        "--detector-batch-size", detector_batch,
        "--recognizer-batch-size", recognizer_batch,
        *extra_args,
        "--model-profile", settings.model_profile_path,
        "--detector-engine", settings.detector_engine_path,
        "--recognizer-engine", settings.recognizer_engine_path,
    ]
    logger.info("[%s] running native worker: %s", WORKER_ID, " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    logger.info("[%s] native worker exited %d", WORKER_ID, proc.returncode)
    if proc.returncode != 0:
        logger.error("native stderr: %s", stderr.decode(errors="ignore")[-2000:])
        raise subprocess.CalledProcessError(proc.returncode or 0, cmd, stdout, stderr)

    if not (output_dir / "manifest.json").exists():
        raise ObservationArtifactError("native worker produced no manifest.json")


def _compress_artifact_bundle(output_dir: Path) -> None:
    """Compress raw protobuf artifacts to zstd and update the manifest.

    The native worker currently emits uncompressed ``.pb`` files; this helper
    publishes them as ``.pb.zst`` and rewrites the manifest SHA-256/size entries.
    It is idempotent: if the compressed files already exist it leaves them.
    """
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.setdefault("artifacts", {})
    compressor = zstandard.ZstdCompressor()

    for uncompressed_name in ("observations.pb", "track_templates.pb"):
        uncompressed = output_dir / uncompressed_name
        compressed_name = uncompressed_name + ".zst"
        compressed = output_dir / compressed_name
        if compressed.exists() or not uncompressed.exists():
            continue
        data = uncompressed.read_bytes()
        compressed.write_bytes(compressor.compress(data))
        uncompressed.unlink()
        artifacts[compressed_name] = {
            "sha256": _sha256_file(compressed),
            "size": compressed.stat().st_size,
        }
        if uncompressed_name in artifacts:
            del artifacts[uncompressed_name]

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _subsample_frames(
    frames: list[VideoObservationFrame],
    job: VideoJob,
    asset: object | None,
) -> list[VideoObservationFrame]:
    if not frames:
        return frames

    mode = job.sampling_mode
    if mode == "every_frame":
        return frames

    sorted_frames = sorted(frames, key=lambda f: (f.pts_ns, f.frame_index))
    kept: list[VideoObservationFrame] = []

    if mode == "every_n_frames" and job.every_n_frames:
        stride = max(1, job.every_n_frames)
        kept = [f for f in sorted_frames if f.frame_index % stride == 0]
    elif mode == "frames_per_second" and job.frames_per_second:
        target_fps = float(job.frames_per_second)
        if target_fps > 0:
            step_ns = 1_000_000_000 / target_fps
            last_ns = -step_ns
            for f in sorted_frames:
                if f.pts_ns - last_ns >= step_ns:
                    kept.append(f)
                    last_ns = f.pts_ns
    else:
        return frames

    return [dataclasses.replace(f, frame_index=i) for i, f in enumerate(kept)]


async def _get_video_asset(video_id: uuid.UUID) -> object:
    async with _uow_factory() as uow:
        asset = await uow.video_assets.get_by_id(video_id)
        if asset is None:
            raise ObservationArtifactError(f"video asset {video_id} not found")
        return asset


async def _process_one_job() -> bool:
    lease_token = generate_uuid7()
    now = datetime.now(UTC)
    lease_expires_at = now + timedelta(seconds=LEASE_SECONDS)

    async with _uow_factory() as uow:
        claimed = await uow.video_job_queue.claim_next(
            worker_id=WORKER_ID,
            lease_token=lease_token,
            now=now,
            lease_expires_at=lease_expires_at,
        )
        if claimed is None:
            return False
        job = claimed.job
        await uow.commit()
        logger.info("[%s] claimed job %s video %s", WORKER_ID, job.job_id, job.video_id)

    object_store = MinIOObjectStore()
    work_dir = Path(tempfile.mkdtemp(prefix=f"mv-worker-{WORKER_ID}-"))
    try:
        async with _uow_factory() as uow:
            current = await uow.video_jobs.get_by_id(JobId(job.job_id))
            if current is not None and current.cancellation_requested:
                await uow.video_job_queue.mark_cancelled(
                    job_id=JobId(job.job_id),
                    worker_id=WORKER_ID,
                    lease_token=lease_token,
                )
                await uow.commit()
                logger.info("[%s] job %s was cancelled before processing", WORKER_ID, job.job_id)
                return True

        asset = await _get_video_asset(job.video_id)
        video_path = await _download_source_video(object_store, job.video_id, work_dir)
        output_dir = work_dir / "out"
        output_dir.mkdir()

        await _run_native_worker(video_path, output_dir, job)
        try:
            bundle = NativeBundle(output_dir)
        except NativeBundleError as exc:
            raise ObservationArtifactError(str(exc)) from exc

        frames = read_observation_frames(bundle.observation_path)
        frames = _subsample_frames(frames, job, asset)
        logger.info("[%s] read %d observation frames (%s)", WORKER_ID, len(frames), job.sampling_mode)

        _compress_artifact_bundle(output_dir)

        processing_service = build_video_processing_service(
            NativeRepresentativeCropProvider(output_dir)
        )
        await processing_service.process(JobId(job.job_id), frames)
        logger.info("[%s] completed job %s", WORKER_ID, job.job_id)
        return True
    except Exception:
        logger.exception("[%s] job %s failed", WORKER_ID, job.job_id)
        async with _uow_factory() as uow:
            current = await uow.video_jobs.get_by_id(JobId(job.job_id))
            if current is not None:
                if current.attempt_count >= current.max_attempts:
                    await uow.video_job_queue.mark_failed(
                        job_id=JobId(job.job_id),
                        worker_id=WORKER_ID,
                        lease_token=lease_token,
                        error_code="WORKER_EXECUTION_FAILED",
                    )
                else:
                    await uow.video_job_queue.release_for_retry(
                        job_id=JobId(job.job_id),
                        worker_id=WORKER_ID,
                        lease_token=lease_token,
                        available_at=datetime.now(UTC) + timedelta(seconds=30),
                        error_code="WORKER_EXECUTION_FAILED",
                    )
                await uow.commit()
        return True
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


async def main() -> None:
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    logger.info("[%s] starting GPU video worker on GPU %s", WORKER_ID, GPU_DEVICE_ID)
    while True:
        try:
            processed = await _process_one_job()
            await asyncio.sleep(0.5 if not processed else 0.1)
        except Exception:
            logger.exception("[%s] worker loop error", WORKER_ID)
            await asyncio.sleep(5.0)


if __name__ == "__main__":
    asyncio.run(main())

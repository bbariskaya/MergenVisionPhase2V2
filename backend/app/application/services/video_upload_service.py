"""Video upload, finalization and async job creation service."""

from __future__ import annotations

import hashlib
import logging
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from app.application.ports.id_generator import IdGenerator
from app.application.ports.object_store import ObjectStore
from app.application.ports.unit_of_work import UnitOfWorkFactory
from app.application.services.identity_storage_lifecycle_service import (
    IdentityStorageLifecycleService,
)
from app.application.services.video_probe_service import ProbeResult, probe_video
from app.domain.entities.process_record import ProcessRecord
from app.domain.entities.video_asset import VideoAsset
from app.domain.entities.video_job import VideoJob
from app.domain.errors import (
    IdempotencyConflictError,
    InvalidMediaError,
    JobNotFoundError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
    ValidationError,
    VideoNotFoundError,
)
from app.domain.value_objects import JobId, ProcessId, UploadSessionId, VideoId

logger = logging.getLogger(__name__)


class FileStream(Protocol):
    filename: str | None
    content_type: str | None

    async def read(self, size: int) -> bytes: ...


@dataclass(frozen=True)
class SubmitResult:
    request_id: str
    process_id: str
    video_id: str
    job_id: str
    upload_session_id: str
    status: str
    status_url: str
    result_url: str

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "requestId": self.request_id,
            "processId": self.process_id,
            "videoId": self.video_id,
            "jobId": self.job_id,
            "uploadSessionId": self.upload_session_id,
            "status": self.status,
            "statusUrl": self.status_url,
            "resultUrl": self.result_url,
        }


@dataclass(frozen=True)
class JobSnapshot:
    request_id: str
    process_id: str
    video_id: str
    job_id: str
    state: str
    stage: str
    progress_percent: int
    sampling_mode: str
    every_n_frames: int | None
    frames_per_second: Decimal | None
    processed_frames: int
    sampled_frames: int
    detected_observations: int
    person_count: int
    cancellation_requested: bool
    error_code: str | None
    status_url: str
    result_url: str


@dataclass(frozen=True)
class VideoSnapshot:
    video_id: str
    upload_session_id: str
    state: str
    content_sha256: str | None
    size_bytes: int | None
    container_format: str | None
    video_codec: str | None
    display_width: int | None
    display_height: int | None
    rotation_degrees: int
    duration_ns: int | None
    total_frames: int | None
    failure_code: str | None


class VideoUploadService:
    def __init__(
        self,
        unit_of_work_factory: UnitOfWorkFactory,
        object_store: ObjectStore,
        lifecycle_service: IdentityStorageLifecycleService,
        id_generator: IdGenerator,
        *,
        bucket_name: str,
        ffprobe_command: list[str],
        max_video_bytes: int,
        max_duration_ns: int,
        max_display_width: int,
        max_display_height: int,
        allowed_containers: set[str],
        allowed_codecs: set[str],
        retention_seconds: int,
        staging_prefix: str,
        source_prefix: str,
        temp_dir: str | None,
        probe_timeout_seconds: float,
        max_attempts: int,
    ) -> None:
        self._uow_factory = unit_of_work_factory
        self._object_store = object_store
        self._lifecycle = lifecycle_service
        self._id_generator = id_generator
        self._bucket_name = bucket_name
        self._ffprobe_command = ffprobe_command
        self._max_video_bytes = max_video_bytes
        self._max_duration_ns = max_duration_ns
        self._max_display_width = max_display_width
        self._max_display_height = max_display_height
        self._allowed_containers = {c.lower() for c in allowed_containers}
        self._allowed_codecs = {c.lower() for c in allowed_codecs}
        self._retention_seconds = retention_seconds
        self._staging_prefix = staging_prefix.rstrip("/") + "/"
        self._source_prefix = source_prefix.rstrip("/") + "/"
        self._temp_dir = temp_dir
        self._probe_timeout_seconds = probe_timeout_seconds
        self._max_attempts = max_attempts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def submit_video_recognition(
        self,
        request_id: str,
        idempotency_key: str,
        file: FileStream,
        *,
        sampling_mode: str = "every_frame",
        every_n_frames: int | None = None,
        frames_per_second: Decimal | float | None = None,
    ) -> SubmitResult:
        self._validate_idempotency_key(idempotency_key)
        self._validate_sampling(sampling_mode, every_n_frames, frames_per_second)

        key_hash = self._hash(idempotency_key)
        request_hash = self._request_hash(
            idempotency_key, sampling_mode, every_n_frames, frames_per_second
        )
        scope = "video_recognize"

        async with self._uow_factory() as uow:
            existing = await uow.idempotency.get(scope, key_hash)
            if existing is not None:
                if existing["request_hash"] != request_hash:
                    raise IdempotencyConflictError(
                        "Idempotency key reused with a different request"
                    )
                if existing["state"] == "completed":
                    snapshot = dict(existing["response_snapshot"] or {})
                    return self._submit_result_from_snapshot(snapshot)
                raise IdempotencyConflictError(
                    "A request with this idempotency key is already in progress"
                )

            expires_at = datetime.now(UTC) + timedelta(hours=24)
            await uow.idempotency.upsert_in_progress(
                scope, key_hash, request_hash, expires_at
            )
            await uow.commit()

        process = await self._lifecycle.start_process(
            "video_recognize",
            {"request_id": request_id},
        )

        try:
            return await self._ingest(
                request_id=request_id,
                process=process,
                file=file,
                scope=scope,
                key_hash=key_hash,
                sampling_mode=sampling_mode,
                every_n_frames=every_n_frames,
                frames_per_second=self._to_decimal(frames_per_second),
            )
        except Exception as exc:
            process_id = str(process.process_id)
            error_code = self._error_code_for(exc)
            logger.exception(
                "Video submit failed for process %s: %s",
                process_id,
                error_code,
            )
            await self._lifecycle.fail_process(process_id=process.process_id, error_code=error_code)
            await self._finalize_idempotency(
                scope=scope,
                key_hash=key_hash,
                state="failed",
                resource_type="process",
                resource_id=process.process_id,
                response_status=400,
                snapshot={
                    "requestId": request_id,
                    "processId": process_id,
                    "status": "failed",
                    "errorCode": error_code,
                },
            )
            raise

    async def get_video(self, video_id_str: str, request_id: str) -> VideoSnapshot:
        try:
            video_id = VideoId(UUID(video_id_str))
        except ValueError as exc:
            raise ValidationError("video_id must be a valid UUID") from exc

        async with self._uow_factory() as uow:
            asset = await uow.video_assets.get_by_id(video_id)
            if asset is None:
                raise VideoNotFoundError(f"Video {video_id_str} not found")
            return self._video_snapshot(request_id, asset)

    async def get_job(self, job_id_str: str, request_id: str) -> JobSnapshot:
        try:
            job_id = JobId(UUID(job_id_str))
        except ValueError as exc:
            raise ValidationError("job_id must be a valid UUID") from exc

        async with self._uow_factory() as uow:
            job = await uow.video_jobs.get_by_id(job_id)
            if job is None:
                raise JobNotFoundError(f"Job {job_id_str} not found")
            return self._job_snapshot(request_id, job)

    async def cancel_job(self, job_id_str: str, request_id: str) -> JobSnapshot:
        try:
            job_id = JobId(UUID(job_id_str))
        except ValueError as exc:
            raise ValidationError("job_id must be a valid UUID") from exc

        async with self._uow_factory() as uow:
            job = await uow.video_jobs.get_by_id(job_id)
            if job is None:
                raise JobNotFoundError(f"Job {job_id_str} not found")

            previous_state = job.state
            job.request_cancellation()
            await uow.video_jobs.update(job)

            if previous_state == "pending":
                process = await uow.processes.get_by_id(job.process_id)
                if process is not None:
                    process.cancel()
                    await uow.processes.update(process)

            await uow.commit()
            return self._job_snapshot(request_id, job)

    async def retry_job(
        self,
        job_id_str: str,
        idempotency_key: str,
        request_id: str,
    ) -> SubmitResult:
        self._validate_idempotency_key(idempotency_key)
        try:
            job_id = JobId(UUID(job_id_str))
        except ValueError as exc:
            raise ValidationError("job_id must be a valid UUID") from exc

        key_hash = self._hash(idempotency_key)
        request_hash = self._hash(f"retry:{idempotency_key}:{job_id_str}")
        scope = "video_retry"

        async with self._uow_factory() as uow:
            existing = await uow.idempotency.get(scope, key_hash)
            if existing is not None:
                if existing["state"] == "completed":
                    snapshot = dict(existing["response_snapshot"] or {})
                    return self._submit_result_from_snapshot(snapshot)
                if existing["request_hash"] != request_hash:
                    raise IdempotencyConflictError(
                        "Idempotency key reused with a different request"
                    )
                raise IdempotencyConflictError(
                    "A retry request with this idempotency key is already in progress"
                )

            original = await uow.video_jobs.get_by_id(job_id)
            if original is None:
                raise JobNotFoundError(f"Job {job_id_str} not found")
            if original.state not in ("failed", "cancelled"):
                raise ValidationError(
                    f"Cannot retry job in state {original.state}; must be failed or cancelled"
                )

            asset = await uow.video_assets.get_by_id(original.video_id)
            if asset is None or asset.state != "ready":
                raise ValidationError("Original video asset is not available for retry")

            expires_at = datetime.now(UTC) + timedelta(hours=24)
            await uow.idempotency.upsert_in_progress(
                scope, key_hash, request_hash, expires_at
            )
            await uow.commit()

        process = await self._lifecycle.start_process(
            "video_recognize",
            {"request_id": request_id, "retry_of_job_id": job_id_str},
        )

        try:
            job = self._create_job(
                video_id=original.video_id,
                process_id=process.process_id,
                sampling_mode=original.sampling_mode,
                every_n_frames=original.every_n_frames,
                frames_per_second=original.frames_per_second,
                retry_of_job_id=job_id,
            )
            async with self._uow_factory() as uow:
                await uow.video_jobs.add(job)
                process_loaded = await uow.processes.get_by_id(process.process_id)
                if process_loaded is None:
                    raise RuntimeError("Process disappeared during retry")
                process_loaded.set_details(
                    {
                        "request_id": request_id,
                        "video_id": str(original.video_id),
                        "job_id": str(job.job_id),
                        "retry_of_job_id": job_id_str,
                        "sampling_mode": job.sampling_mode,
                    }
                )
                await uow.processes.update(process_loaded)
                await uow.commit()

            result = self._submit_result(request_id, process, original.video_id, job, None)
            await self._finalize_idempotency(
                scope=scope,
                key_hash=key_hash,
                state="completed",
                resource_type="job",
                resource_id=job.job_id,
                response_status=202,
                snapshot=result.to_snapshot(),
            )
            return result
        except Exception:
            await self._lifecycle.fail_process(
                process_id=process.process_id, error_code="retry_creation_failed"
            )
            await self._finalize_idempotency(
                scope=scope,
                key_hash=key_hash,
                state="failed",
                resource_type="process",
                resource_id=process.process_id,
                response_status=500,
                snapshot={
                    "requestId": request_id,
                    "processId": str(process.process_id),
                    "status": "failed",
                    "errorCode": "retry_creation_failed",
                },
            )
            raise

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    async def _ingest(
        self,
        *,
        request_id: str,
        process: ProcessRecord,
        file: FileStream,
        scope: str,
        key_hash: str,
        sampling_mode: str,
        every_n_frames: int | None,
        frames_per_second: Decimal | None,
    ) -> SubmitResult:
        video_id = VideoId(self._id_generator.new_uuid7())
        upload_session_id = UploadSessionId(self._id_generator.new_uuid7())
        asset = VideoAsset(
            video_id=video_id,
            upload_session_id=upload_session_id,
            state="uploading",
            staging_bucket=self._bucket_name,
            staging_object_key=f"{self._staging_prefix}{video_id}/upload",
            content_type=file.content_type,
        )

        staging_key = asset.staging_object_key
        if staging_key is None:
            raise RuntimeError("Staging object key not set")
        canonical_key = f"{self._source_prefix}{video_id}/source/original"

        temp_path = await self._stream_to_temp(file, asset)
        probe = await probe_video(
            path=temp_path,
            ffprobe_command=self._ffprobe_command,
            timeout_seconds=self._probe_timeout_seconds,
            max_duration_ns=self._max_duration_ns,
        )

        self._validate_probe(probe)

        await self._object_store.upload_from_file(
            key=staging_key,
            file_path=temp_path,
            content_type=asset.content_type or "video/mp4",
        )

        await self._object_store.copy(staging_key, canonical_key)
        canonical_stat = await self._object_store.stat(canonical_key)
        if canonical_stat is None:
            raise RuntimeError("Canonical video object missing after finalize copy")
        if canonical_stat.sha256 is None:
            raise RuntimeError("Canonical video object missing SHA-256 metadata")

        retention_until = datetime.now(UTC) + timedelta(seconds=self._retention_seconds)
        asset.mark_ready(
            bucket=self._bucket_name,
            object_key=canonical_key,
            content_sha256=canonical_stat.sha256,
            size_bytes=canonical_stat.size,
            content_type=asset.content_type or "video/mp4",
            container_format=probe.container_format,
            video_codec=probe.video_codec,
            pixel_format=probe.pixel_format,
            display_width=probe.display_width,
            display_height=probe.display_height,
            rotation_degrees=probe.rotation_degrees,
            duration_ns=probe.duration_ns,
            time_base_num=probe.time_base_num,
            time_base_den=probe.time_base_den,
            nominal_fps_num=probe.nominal_fps_num,
            nominal_fps_den=probe.nominal_fps_den,
            total_frames=probe.total_frames,
            retention_until=retention_until,
        )

        job = self._create_job(
            video_id=video_id,
            process_id=process.process_id,
            sampling_mode=sampling_mode,
            every_n_frames=every_n_frames,
            frames_per_second=frames_per_second,
        )

        async with self._uow_factory() as uow:
            await uow.video_assets.add(asset)
            await uow.video_jobs.add(job)
            process_loaded = await uow.processes.get_by_id(process.process_id)
            if process_loaded is None:
                raise RuntimeError("Process disappeared during video ingest")
            process_loaded.set_details(
                {
                    "request_id": request_id,
                    "video_id": str(video_id),
                    "job_id": str(job.job_id),
                    "upload_session_id": str(upload_session_id),
                    "sampling_mode": sampling_mode,
                }
            )
            await uow.processes.update(process_loaded)
            await uow.commit()

        result = self._submit_result(request_id, process, video_id, job, asset.upload_session_id)
        await self._finalize_idempotency(
            scope=scope,
            key_hash=key_hash,
            state="completed",
            resource_type="job",
            resource_id=job.job_id,
            response_status=202,
            snapshot=result.to_snapshot(),
        )

        try:
            await self._object_store.delete(staging_key)
        except Exception:
            logger.exception("Staging cleanup failed for %s", staging_key)

        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            logger.exception("Temp file cleanup failed for %s", temp_path)

        return result

    async def _stream_to_temp(self, file: FileStream, asset: VideoAsset) -> Path:
        temp = tempfile.NamedTemporaryFile(
            delete=False,
            dir=self._temp_dir,
            prefix="mv-upload-",
            suffix=".bin",
        )
        temp_path = Path(temp.name)
        sha = hashlib.sha256()
        total = 0
        limit = self._max_video_bytes + 1

        try:
            while limit > 0:
                chunk = await file.read(min(limit, 65536))
                if not chunk:
                    break
                temp.write(chunk)
                sha.update(chunk)
                total += len(chunk)
                limit -= len(chunk)

            if limit == 0:
                raise PayloadTooLargeError(
                    f"Video exceeds maximum allowed size of {self._max_video_bytes} bytes"
                )
        finally:
            temp.close()

        asset.size_bytes = total
        asset.content_sha256 = sha.hexdigest()
        return temp_path

    def _create_job(
        self,
        *,
        video_id: VideoId,
        process_id: ProcessId,
        sampling_mode: str,
        every_n_frames: int | None,
        frames_per_second: Decimal | None,
        retry_of_job_id: JobId | None = None,
    ) -> VideoJob:
        return VideoJob(
            job_id=JobId(self._id_generator.new_uuid7()),
            video_id=video_id,
            process_id=process_id,
            retry_of_job_id=retry_of_job_id,
            state="pending",
            stage="queued",
            sampling_mode=sampling_mode,
            every_n_frames=every_n_frames,
            frames_per_second=frames_per_second,
            max_attempts=self._max_attempts,
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate_idempotency_key(self, value: str) -> None:
        if not isinstance(value, str) or len(value) == 0 or len(value) > 1024:
            raise ValidationError("Idempotency-Key is required and must be <= 1024 chars")

    def _validate_sampling(
        self,
        mode: str,
        every_n_frames: int | None,
        frames_per_second: Decimal | float | None,
    ) -> None:
        if mode == "every_frame":
            if every_n_frames is not None or frames_per_second is not None:
                raise ValidationError("every_frame sampling takes no extra parameters")
        elif mode == "every_n_frames":
            if every_n_frames is None or every_n_frames <= 0 or frames_per_second is not None:
                raise ValidationError("every_n_frames requires positive integer every_n_frames")
        elif mode == "frames_per_second":
            fps = self._to_decimal(frames_per_second)
            if fps is None or fps <= 0 or every_n_frames is not None:
                raise ValidationError(
                    "frames_per_second requires positive decimal frames_per_second"
                )
        else:
            raise ValidationError(
                "sampling_mode must be one of every_frame, every_n_frames, frames_per_second"
            )

    def _validate_probe(self, probe: ProbeResult) -> None:
        if probe.container_format not in self._allowed_containers:
            raise UnsupportedMediaTypeError(
                f"Container '{probe.container_format}' is not supported"
            )
        if probe.video_codec not in self._allowed_codecs:
            raise UnsupportedMediaTypeError(
                f"Video codec '{probe.video_codec}' is not supported"
            )
        if probe.display_width > self._max_display_width:
            raise InvalidMediaError("Video width exceeds maximum allowed")
        if probe.display_height > self._max_display_height:
            raise InvalidMediaError("Video height exceeds maximum allowed")

    # ------------------------------------------------------------------
    # Idempotency helpers
    # ------------------------------------------------------------------
    def _hash(self, value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _request_hash(
        self,
        idempotency_key: str,
        sampling_mode: str,
        every_n_frames: int | None,
        frames_per_second: Decimal | float | None,
    ) -> str:
        parts = [
            idempotency_key,
            sampling_mode,
            str(every_n_frames),
            str(frames_per_second),
        ]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

    async def _finalize_idempotency(
        self,
        *,
        scope: str,
        key_hash: str,
        state: str,
        resource_type: str,
        resource_id: UUID,
        response_status: int,
        snapshot: dict[str, Any],
    ) -> None:
        async with self._uow_factory() as uow:
            await uow.idempotency.finalize(
                scope=scope,
                key_hash=key_hash,
                state=state,
                resource_type=resource_type,
                resource_id=resource_id,
                response_status=response_status,
                response_snapshot=snapshot,
            )
            await uow.commit()

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------
    def _submit_result(
        self,
        request_id: str,
        process: ProcessRecord,
        video_id: VideoId,
        job: VideoJob,
        upload_session_id: UploadSessionId | None,
    ) -> SubmitResult:
        job_id_str = str(job.job_id)
        return SubmitResult(
            request_id=request_id,
            process_id=str(process.process_id),
            video_id=str(video_id),
            job_id=job_id_str,
            upload_session_id=str(upload_session_id) if upload_session_id else "",
            status=job.state,
            status_url=f"/api/v1/videos/jobs/{job_id_str}",
            result_url=f"/api/v1/videos/jobs/{job_id_str}/result",
        )

    def _submit_result_from_snapshot(self, snapshot: dict[str, Any]) -> SubmitResult:
        return SubmitResult(
            request_id=snapshot["requestId"],
            process_id=snapshot["processId"],
            video_id=snapshot["videoId"],
            job_id=snapshot["jobId"],
            upload_session_id=snapshot.get("uploadSessionId", ""),
            status=snapshot["status"],
            status_url=snapshot["statusUrl"],
            result_url=snapshot["resultUrl"],
        )

    def _video_snapshot(self, request_id: str, asset: VideoAsset) -> VideoSnapshot:
        return VideoSnapshot(
            video_id=str(asset.video_id),
            upload_session_id=str(asset.upload_session_id),
            state=asset.state,
            content_sha256=asset.content_sha256,
            size_bytes=asset.size_bytes,
            container_format=asset.container_format,
            video_codec=asset.video_codec,
            display_width=asset.display_width,
            display_height=asset.display_height,
            rotation_degrees=asset.rotation_degrees,
            duration_ns=asset.duration_ns,
            total_frames=asset.total_frames,
            failure_code=asset.failure_code,
        )

    def _job_snapshot(self, request_id: str, job: VideoJob) -> JobSnapshot:
        job_id_str = str(job.job_id)
        return JobSnapshot(
            request_id=request_id,
            process_id=str(job.process_id),
            video_id=str(job.video_id),
            job_id=job_id_str,
            state=job.state,
            stage=job.stage,
            progress_percent=job.progress_percent,
            sampling_mode=job.sampling_mode,
            every_n_frames=job.every_n_frames,
            frames_per_second=job.frames_per_second,
            processed_frames=job.processed_frames,
            sampled_frames=job.sampled_frames,
            detected_observations=job.detected_observations,
            person_count=job.person_count,
            cancellation_requested=job.cancellation_requested,
            error_code=job.error_code,
            status_url=f"/api/v1/videos/jobs/{job_id_str}",
            result_url=f"/api/v1/videos/jobs/{job_id_str}/result",
        )

    def _to_decimal(self, value: Decimal | float | None) -> Decimal | None:
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))

    def _error_code_for(self, exc: BaseException) -> str:
        if isinstance(exc, PayloadTooLargeError):
            return "payload_too_large"
        if isinstance(exc, UnsupportedMediaTypeError):
            return "unsupported_media_type"
        if isinstance(exc, InvalidMediaError):
            return "invalid_media"
        if isinstance(exc, IdempotencyConflictError):
            return "idempotency_conflict"
        return "upload_failed"


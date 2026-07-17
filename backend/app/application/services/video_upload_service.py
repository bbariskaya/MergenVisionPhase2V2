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
from app.application.services.video_probe_service import ProbeResult, probe_video
from app.domain.entities.outbox_event import OutboxEvent
from app.domain.entities.process_event import ProcessEvent
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


def _sanitize_error_code(code: str) -> str:
    """Keep internal probe messages out of API responses."""
    public_codes = {
        "payload_too_large",
        "unsupported_media_type",
        "invalid_media",
        "upload_failed",
        "idempotency_conflict",
        "video_probe_timeout",
        "video_probe_failed",
        "minio_upload_failed",
        "minio_copy_failed",
        "canonical_object_missing",
        "canonical_checksum_mismatch",
        "retry_creation_failed",
        "resource_not_found",
    }
    return code if code in public_codes else "upload_failed"


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


@dataclass(frozen=True)
class JobResultSnapshot:
    request_id: str
    job_id: str
    state: str
    result_available: bool
    manifest_bucket: str | None
    manifest_key: str | None
    manifest_sha256: str | None


class VideoUploadService:
    def __init__(
        self,
        unit_of_work_factory: UnitOfWorkFactory,
        object_store: ObjectStore,
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
        outbox_max_attempts: int = 3,
    ) -> None:
        self._uow_factory = unit_of_work_factory
        self._object_store = object_store
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
        self._outbox_max_attempts = outbox_max_attempts

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
        self._validate_sampling(
            sampling_mode,
            every_n_frames,
            self._to_decimal(frames_per_second),
        )

        # Validation 1: stream to a bounded temp file and compute SHA-256.
        temp_path = await self._stream_to_temp(file)
        content_sha256 = self._sha256_of_file(temp_path)
        size_bytes = temp_path.stat().st_size

        key_hash = self._hash(idempotency_key)
        request_hash = self._request_hash(
            idempotency_key,
            sampling_mode,
            self._to_decimal(every_n_frames),
            self._to_decimal(frames_per_second),
            content_sha256,
        )
        scope = "video_recognize"

        # Transaction 1: atomic idempotency claim + durable reservation.
        # No external storage calls happen before this commits.
        async with self._uow_factory() as uow:
            existing = await uow.idempotency.claim(
                scope=scope,
                key_hash=key_hash,
                request_hash=request_hash,
                expires_at=datetime.now(UTC) + timedelta(hours=24),
            )
            if existing is not None:
                return await self._handle_existing_idempotency(existing, request_hash)

            video_id = VideoId(self._id_generator.new_uuid7())
            upload_session_id = UploadSessionId(self._id_generator.new_uuid7())
            process_id = ProcessId(self._id_generator.new_uuid7())

            process = ProcessRecord(
                process_id=process_id,
                process_type="video_recognize",
                status="processing",
                details={
                    "request_id": request_id,
                    "upload_session_id": str(upload_session_id),
                    "video_id": str(video_id),
                },
            )

            asset = VideoAsset(
                video_id=video_id,
                upload_session_id=upload_session_id,
                state="uploading",
                staging_bucket=self._bucket_name,
                staging_object_key=f"{self._staging_prefix}{video_id}/upload",
                content_type=file.content_type,
                content_sha256=content_sha256,
                size_bytes=size_bytes,
            )

            await uow.processes.add(process)
            await uow.video_assets.add(asset)

            sequence = await uow.process_events.next_sequence(str(process_id))
            await uow.process_events.add(
                ProcessEvent(
                    event_id=self._id_generator.new_uuid7(),
                    process_id=process_id,
                    sequence_no=sequence,
                    event_type="upload_reserved",
                    severity="info",
                    payload={
                        "video_id": str(video_id),
                        "upload_session_id": str(upload_session_id),
                        "size_bytes": size_bytes,
                        "content_sha256": content_sha256,
                        "content_type": file.content_type,
                        "sampling_mode": sampling_mode,
                        "every_n_frames": every_n_frames,
                        "frames_per_second": self._to_decimal(frames_per_second),
                    },
                )
            )

            await uow.commit()

        # External operations: probe, upload to staging, canonical copy, verify.
        try:
            probe = await probe_video(
                path=temp_path,
                ffprobe_command=self._ffprobe_command,
                timeout_seconds=self._probe_timeout_seconds,
                max_duration_ns=self._max_duration_ns,
            )
            self._validate_probe(probe)

            staging_key = asset.staging_object_key
            if staging_key is None:
                raise RuntimeError("Staging object key not set")
            canonical_key = f"{self._source_prefix}{video_id}/source/original"

            staging_stat = await self._object_store.upload_from_file(
                key=staging_key,
                file_path=temp_path,
                content_type=self._canonical_content_type(probe, file.content_type),
            )

            if staging_stat.sha256 != content_sha256 or staging_stat.size != size_bytes:
                raise InvalidMediaError("canonical_checksum_mismatch")

            await self._object_store.copy(staging_key, canonical_key)
            canonical_stat = await self._object_store.stat(canonical_key)
            if canonical_stat is None:
                raise InvalidMediaError("canonical_object_missing")
            if canonical_stat.sha256 != content_sha256 or canonical_stat.size != size_bytes:
                raise InvalidMediaError("canonical_checksum_mismatch")

            every_n_decimal = self._to_decimal(every_n_frames)
            frames_per_decimal = self._to_decimal(frames_per_second)

            job = self._create_job(
                video_id=video_id,
                process_id=process_id,
                sampling_mode=sampling_mode,
                every_n_frames=every_n_decimal,
                frames_per_second=frames_per_decimal,
            )

            # Transaction 2: finalize asset, create job, complete reservation.
            async with self._uow_factory() as uow:
                loaded_asset = await uow.video_assets.get_by_id(video_id)
                if loaded_asset is None:
                    raise RuntimeError("Video asset disappeared during ingest")
                loaded_asset.mark_ready(
                    bucket=self._bucket_name,
                    object_key=canonical_key,
                    content_sha256=canonical_stat.sha256,
                    size_bytes=canonical_stat.size,
                    content_type=self._canonical_content_type(probe, file.content_type),
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
                    retention_until=datetime.now(UTC)
                    + timedelta(seconds=self._retention_seconds),
                )
                await uow.video_assets.update(loaded_asset)

                loaded_process = await uow.processes.get_by_id(process_id)
                if loaded_process is None:
                    raise RuntimeError("Process disappeared during ingest")
                loaded_process.set_details(
                    {
                        "request_id": request_id,
                        "video_id": str(video_id),
                        "job_id": str(job.job_id),
                        "upload_session_id": str(upload_session_id),
                        "sampling_mode": sampling_mode,
                        "content_sha256": canonical_stat.sha256,
                    }
                )
                await uow.processes.update(loaded_process)

                await uow.video_jobs.add(job)

                seq = await uow.process_events.next_sequence(str(process_id))
                await uow.process_events.add(
                    ProcessEvent(
                        event_id=self._id_generator.new_uuid7(),
                        process_id=process_id,
                        job_id=job.job_id,
                        sequence_no=seq,
                        event_type="video_ready",
                        severity="info",
                        payload={
                            "video_id": str(video_id),
                            "canonical_key": canonical_key,
                            "content_sha256": canonical_stat.sha256,
                            "size_bytes": canonical_stat.size,
                        },
                    )
                )
                seq += 1
                await uow.process_events.add(
                    ProcessEvent(
                        event_id=self._id_generator.new_uuid7(),
                        process_id=process_id,
                        job_id=job.job_id,
                        sequence_no=seq,
                        event_type="job_created",
                        severity="info",
                        payload={
                            "job_id": str(job.job_id),
                            "sampling_mode": sampling_mode,
                            "every_n_frames": every_n_decimal,
                            "frames_per_second": frames_per_decimal,
                        },
                    )
                )

                await uow.outbox.add(
                    OutboxEvent(
                        outbox_event_id=self._id_generator.new_uuid7(),
                        aggregate_type="video_asset",
                        aggregate_id=video_id,
                        event_type="cleanup_staging",
                        dedupe_key=f"video-asset:{video_id}:cleanup-staging",
                        state="pending",
                        attempt_count=0,
                        max_attempts=self._outbox_max_attempts,
                        available_at=datetime.now(UTC),
                        payload={"bucket": self._bucket_name, "object_key": staging_key},
                    )
                )

                result = self._submit_result(
                    request_id, loaded_process, video_id, job, upload_session_id
                )
                await uow.idempotency.finalize(
                    scope=scope,
                    key_hash=key_hash,
                    state="completed",
                    resource_type="job",
                    resource_id=job.job_id,
                    response_status=202,
                    response_snapshot=result.to_snapshot(),
                )
                await uow.commit()

            # Best-effort staging cleanup; outbox guarantees eventual consistency.
            await self._delete_best_effort(staging_key)
            await self._unlink_best_effort(temp_path)
            return result

        except Exception as exc:
            error_code = self._error_code_for(exc)
            await self._fail_ingest(
                request_id=request_id,
                scope=scope,
                key_hash=key_hash,
                video_id=video_id,
                process_id=process_id,
                staging_key=asset.staging_object_key,
                temp_path=temp_path,
                error_code=error_code,
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

    async def get_job_result(
        self, job_id_str: str, request_id: str
    ) -> JobResultSnapshot:
        try:
            job_id = JobId(UUID(job_id_str))
        except ValueError as exc:
            raise ValidationError("job_id must be a valid UUID") from exc

        async with self._uow_factory() as uow:
            job = await uow.video_jobs.get_by_id(job_id)
            if job is None:
                raise JobNotFoundError(f"Job {job_id_str} not found")

            return JobResultSnapshot(
                request_id=request_id,
                job_id=job_id_str,
                state=job.state,
                result_available=job.state == "completed"
                and job.result_manifest_key is not None,
                manifest_bucket=job.result_manifest_bucket,
                manifest_key=job.result_manifest_key,
                manifest_sha256=job.result_manifest_sha256,
            )

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
            existing = await uow.idempotency.claim(
                scope=scope,
                key_hash=key_hash,
                request_hash=request_hash,
                expires_at=datetime.now(UTC) + timedelta(hours=24),
            )
            if existing is not None:
                return await self._handle_existing_idempotency(existing, request_hash)

            process_id = ProcessId(self._id_generator.new_uuid7())
            process = ProcessRecord(
                process_id=process_id,
                process_type="video_recognize",
                status="processing",
                details={"request_id": request_id, "retry_of_job_id": job_id_str},
            )
            await uow.processes.add(process)

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

            await uow.idempotency.set_in_progress(
                scope=scope,
                key_hash=key_hash,
                request_hash=request_hash,
                expires_at=datetime.now(UTC) + timedelta(hours=24),
            )
            await uow.commit()

        try:
            job = self._create_job(
                video_id=original.video_id,
                process_id=process_id,
                sampling_mode=original.sampling_mode,
                every_n_frames=original.every_n_frames,
                frames_per_second=original.frames_per_second,
                retry_of_job_id=job_id,
            )
            async with self._uow_factory() as uow:
                await uow.video_jobs.add(job)
                process_loaded = await uow.processes.get_by_id(process_id)
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

                seq = await uow.process_events.next_sequence(str(process_id))
                await uow.process_events.add(
                    ProcessEvent(
                        event_id=self._id_generator.new_uuid7(),
                        process_id=process_id,
                        job_id=job.job_id,
                        sequence_no=seq,
                        event_type="job_created",
                        severity="info",
                        payload={
                            "job_id": str(job.job_id),
                            "retry_of_job_id": job_id_str,
                            "sampling_mode": job.sampling_mode,
                        },
                    )
                )

                result = self._submit_result(request_id, process_loaded, original.video_id, job, None)
                await uow.idempotency.finalize(
                    scope=scope,
                    key_hash=key_hash,
                    state="completed",
                    resource_type="job",
                    resource_id=job.job_id,
                    response_status=202,
                    response_snapshot=result.to_snapshot(),
                )
                await uow.commit()

            return result
        except Exception:
            async with self._uow_factory() as uow:
                loaded = await uow.processes.get_by_id(process_id)
                if loaded is not None and loaded.status == "processing":
                    loaded.fail("retry_creation_failed")
                    await uow.processes.update(loaded)
                await uow.idempotency.finalize(
                    scope=scope,
                    key_hash=key_hash,
                    state="failed",
                    resource_type="process",
                    resource_id=process_id,
                    response_status=500,
                    response_snapshot={
                        "requestId": request_id,
                        "processId": str(process_id),
                        "status": "failed",
                        "errorCode": "retry_creation_failed",
                    },
                )
                await uow.commit()
            raise

    # ------------------------------------------------------------------
    # Failure handling
    # ------------------------------------------------------------------
    async def _fail_ingest(
        self,
        *,
        request_id: str,
        scope: str,
        key_hash: str,
        video_id: VideoId,
        process_id: ProcessId,
        staging_key: str | None,
        temp_path: Path,
        error_code: str,
    ) -> None:
        safe_code = _sanitize_error_code(error_code)
        logger.exception(
            "Video submit failed for process %s video %s: %s",
            process_id,
            video_id,
            safe_code,
        )
        try:
            async with self._uow_factory() as uow:
                asset = await uow.video_assets.get_by_id(video_id)
                if asset is not None and asset.state == "uploading":
                    asset.mark_rejected(safe_code)
                    await uow.video_assets.update(asset)

                process = await uow.processes.get_by_id(process_id)
                if process is not None and process.status == "processing":
                    process.fail(safe_code)
                    await uow.processes.update(process)

                seq = await uow.process_events.next_sequence(str(process_id))
                await uow.process_events.add(
                    ProcessEvent(
                        event_id=self._id_generator.new_uuid7(),
                        process_id=process_id,
                        sequence_no=seq,
                        event_type="upload_failed",
                        severity="error",
                        payload={"error_code": safe_code, "video_id": str(video_id)},
                    )
                )

                if asset is not None and asset.object_key is not None:
                    await uow.outbox.add(
                        OutboxEvent(
                            outbox_event_id=self._id_generator.new_uuid7(),
                            aggregate_type="video_asset",
                            aggregate_id=video_id,
                            event_type="cleanup_canonical",
                            dedupe_key=f"video-asset:{video_id}:cleanup-canonical",
                            state="pending",
                            attempt_count=0,
                            max_attempts=self._outbox_max_attempts,
                            available_at=datetime.now(UTC),
                            payload={
                                "bucket": self._bucket_name,
                                "object_key": asset.object_key,
                            },
                        )
                    )

                snapshot = {
                    "requestId": request_id,
                    "processId": str(process_id),
                    "videoId": str(video_id),
                    "status": "failed",
                    "errorCode": safe_code,
                }
                await uow.idempotency.finalize(
                    scope=scope,
                    key_hash=key_hash,
                    state="failed",
                    resource_type="process",
                    resource_id=process_id,
                    response_status=400,
                    response_snapshot=snapshot,
                )
                await uow.commit()
        finally:
            if staging_key:
                await self._delete_best_effort(staging_key)
            await self._unlink_best_effort(temp_path)

    # ------------------------------------------------------------------
    # Ingestion helpers
    # ------------------------------------------------------------------
    async def _stream_to_temp(self, file: FileStream) -> Path:
        temp = tempfile.NamedTemporaryFile(
            delete=False,
            dir=self._temp_dir,
            prefix="mv-upload-",
            suffix=".bin",
        )
        temp_path = Path(temp.name)
        limit = self._max_video_bytes + 1

        try:
            while limit > 0:
                chunk = await file.read(min(limit, 65536))
                if not chunk:
                    break
                temp.write(chunk)
                limit -= len(chunk)
            if limit == 0:
                raise PayloadTooLargeError(
                    f"Video exceeds maximum allowed size of {self._max_video_bytes} bytes"
                )
        finally:
            temp.close()

        return temp_path

    def _sha256_of_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def _create_job(
        self,
        *,
        video_id: VideoId,
        process_id: ProcessId,
        sampling_mode: str,
        every_n_frames: Decimal | None,
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
            every_n_frames=int(every_n_frames) if every_n_frames is not None else None,
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
        every_n_frames: int | Decimal | None,
        frames_per_second: Decimal | None,
    ) -> None:
        if mode == "every_frame":
            if every_n_frames is not None or frames_per_second is not None:
                raise ValidationError("every_frame sampling takes no extra parameters")
        elif mode == "every_n_frames":
            if every_n_frames is None or int(every_n_frames) <= 0 or frames_per_second is not None:
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

    def _canonical_content_type(self, probe: ProbeResult, fallback: str | None) -> str:
        if probe.container_format == "mp4":
            return "video/mp4"
        if probe.container_format == "mkv":
            return "video/x-matroska"
        return fallback or "video/mp4"

    # ------------------------------------------------------------------
    # Idempotency helpers
    # ------------------------------------------------------------------
    def _hash(self, value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _request_hash(
        self,
        idempotency_key: str,
        sampling_mode: str,
        every_n_frames: Decimal | None,
        frames_per_second: Decimal | None,
        content_sha256: str,
    ) -> str:
        parts = [
            idempotency_key,
            sampling_mode,
            str(every_n_frames),
            str(frames_per_second),
            content_sha256,
        ]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

    async def _handle_existing_idempotency(
        self, existing: dict[str, Any], request_hash: str
    ) -> SubmitResult:
        if existing["request_hash"] != request_hash:
            raise IdempotencyConflictError(
                "Idempotency key reused with a different request"
            )
        if existing["state"] == "completed":
            snapshot = dict(existing["response_snapshot"] or {})
            return self._submit_result_from_snapshot(snapshot)
        if existing["state"] == "failed":
            # Allow retry with the same idempotency key; transaction will reclaim.
            return None  # type: ignore[return-value]
        raise IdempotencyConflictError(
            "A request with this idempotency key is already in progress"
        )

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

    def _to_decimal(self, value: int | Decimal | float | None) -> Decimal | None:
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))

    def _error_code_for(self, exc: BaseException) -> str:
        if isinstance(exc, TimeoutError):
            return "video_probe_timeout"
        if isinstance(exc, PayloadTooLargeError):
            return "payload_too_large"
        if isinstance(exc, UnsupportedMediaTypeError):
            return "unsupported_media_type"
        if isinstance(exc, InvalidMediaError):
            message = str(exc).lower()
            if "timed out" in message:
                return "video_probe_timeout"
            if "probe failed" in message:
                return "video_probe_failed"
            return "invalid_media"
        if isinstance(exc, IdempotencyConflictError):
            return "idempotency_conflict"
        return "upload_failed"

    async def _delete_best_effort(self, key: str) -> None:
        try:
            await self._object_store.delete(key)
        except Exception:
            logger.exception("Object cleanup failed for %s", key)

    async def _unlink_best_effort(self, temp_path: Path) -> None:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            logger.exception("Temp file cleanup failed for %s", temp_path)

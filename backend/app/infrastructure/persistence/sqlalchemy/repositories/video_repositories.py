"""SQLAlchemy repositories for Phase 2 video control plane."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities.video_asset import VideoAsset
from app.domain.entities.video_job import VideoJob
from app.domain.value_objects import JobId, ProcessId, UploadSessionId, VideoId
from app.infrastructure.persistence.sqlalchemy.models.idempotency_record import (
    IdempotencyRecordOrm,
)
from app.infrastructure.persistence.sqlalchemy.models.video_asset import VideoAssetOrm
from app.infrastructure.persistence.sqlalchemy.models.video_job import VideoJobOrm


def _asset_to_domain(orm: VideoAssetOrm) -> VideoAsset:
    return VideoAsset(
        video_id=VideoId(orm.video_id),
        upload_session_id=UploadSessionId(orm.upload_session_id),
        state=orm.state,
        staging_bucket=orm.staging_bucket,
        staging_object_key=orm.staging_object_key,
        bucket=orm.bucket,
        object_key=orm.object_key,
        content_sha256=orm.content_sha256,
        size_bytes=orm.size_bytes,
        content_type=orm.content_type,
        container_format=orm.container_format,
        video_codec=orm.video_codec,
        pixel_format=orm.pixel_format,
        display_width=orm.display_width,
        display_height=orm.display_height,
        rotation_degrees=orm.rotation_degrees,
        duration_ns=orm.duration_ns,
        time_base_num=orm.time_base_num,
        time_base_den=orm.time_base_den,
        nominal_fps_num=orm.nominal_fps_num,
        nominal_fps_den=orm.nominal_fps_den,
        total_frames=orm.total_frames,
        retention_until=orm.retention_until,
        failure_code=orm.failure_code,
        created_at=orm.created_at,
        updated_at=orm.updated_at,
        ready_at=orm.ready_at,
        deleted_at=orm.deleted_at,
        version=orm.version,
    )


def _job_to_domain(orm: VideoJobOrm) -> VideoJob:
    return VideoJob(
        job_id=JobId(orm.job_id),
        video_id=VideoId(orm.video_id),
        process_id=ProcessId(orm.process_id),
        retry_of_job_id=JobId(orm.retry_of_job_id) if orm.retry_of_job_id else None,
        state=orm.state,
        stage=orm.stage,
        progress_percent=orm.progress_percent,
        sampling_mode=orm.sampling_mode,
        every_n_frames=orm.every_n_frames,
        frames_per_second=orm.frames_per_second,
        processed_frames=orm.processed_frames,
        sampled_frames=orm.sampled_frames,
        detected_observations=orm.detected_observations,
        person_count=orm.person_count,
        available_at=orm.available_at,
        lease_owner=orm.lease_owner,
        lease_token=orm.lease_token,
        lease_expires_at=orm.lease_expires_at,
        heartbeat_at=orm.heartbeat_at,
        attempt_count=orm.attempt_count,
        max_attempts=orm.max_attempts,
        cancellation_requested=orm.cancellation_requested,
        error_code=orm.error_code,
        created_at=orm.created_at,
        updated_at=orm.updated_at,
        started_at=orm.started_at,
        completed_at=orm.completed_at,
        failed_at=orm.failed_at,
        cancelled_at=orm.cancelled_at,
        version=orm.version,
        result_manifest_bucket=orm.result_manifest_bucket,
        result_manifest_key=orm.result_manifest_key,
        result_manifest_sha256=orm.result_manifest_sha256,
        result_schema_version=orm.result_schema_version,
    )


class SqlAlchemyVideoAssetRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, asset: VideoAsset) -> None:
        orm = VideoAssetOrm(
            video_id=asset.video_id,
            upload_session_id=asset.upload_session_id,
            state=asset.state,
            staging_bucket=asset.staging_bucket,
            staging_object_key=asset.staging_object_key,
            bucket=asset.bucket,
            object_key=asset.object_key,
            content_sha256=asset.content_sha256,
            size_bytes=asset.size_bytes,
            content_type=asset.content_type,
            container_format=asset.container_format,
            video_codec=asset.video_codec,
            pixel_format=asset.pixel_format,
            display_width=asset.display_width,
            display_height=asset.display_height,
            rotation_degrees=asset.rotation_degrees,
            duration_ns=asset.duration_ns,
            time_base_num=asset.time_base_num,
            time_base_den=asset.time_base_den,
            nominal_fps_num=asset.nominal_fps_num,
            nominal_fps_den=asset.nominal_fps_den,
            total_frames=asset.total_frames,
            retention_until=asset.retention_until,
            failure_code=asset.failure_code,
            created_at=asset.created_at,
            updated_at=asset.updated_at,
            ready_at=asset.ready_at,
            deleted_at=asset.deleted_at,
            version=asset.version,
        )
        self._session.add(orm)

    async def get_by_id(self, video_id: VideoId) -> VideoAsset | None:
        result = await self._session.execute(
            select(VideoAssetOrm).where(VideoAssetOrm.video_id == video_id)
        )
        orm = result.scalar_one_or_none()
        return _asset_to_domain(orm) if orm else None

    async def get_by_upload_session_id(
        self, upload_session_id: UploadSessionId
    ) -> VideoAsset | None:
        result = await self._session.execute(
            select(VideoAssetOrm).where(
                VideoAssetOrm.upload_session_id == upload_session_id
            )
        )
        orm = result.scalar_one_or_none()
        return _asset_to_domain(orm) if orm else None

    async def update(self, asset: VideoAsset) -> None:
        result = await self._session.execute(
            select(VideoAssetOrm).where(VideoAssetOrm.video_id == asset.video_id)
        )
        orm = result.scalar_one_or_none()
        if orm is None:
            raise ValueError(f"VideoAsset {asset.video_id} not found")
        orm.state = asset.state
        orm.staging_bucket = asset.staging_bucket
        orm.staging_object_key = asset.staging_object_key
        orm.bucket = asset.bucket
        orm.object_key = asset.object_key
        orm.content_sha256 = asset.content_sha256
        orm.size_bytes = asset.size_bytes
        orm.content_type = asset.content_type
        orm.container_format = asset.container_format
        orm.video_codec = asset.video_codec
        orm.pixel_format = asset.pixel_format
        orm.display_width = asset.display_width
        orm.display_height = asset.display_height
        orm.rotation_degrees = asset.rotation_degrees
        orm.duration_ns = asset.duration_ns
        orm.time_base_num = asset.time_base_num
        orm.time_base_den = asset.time_base_den
        orm.nominal_fps_num = asset.nominal_fps_num
        orm.nominal_fps_den = asset.nominal_fps_den
        orm.total_frames = asset.total_frames
        orm.retention_until = asset.retention_until
        orm.failure_code = asset.failure_code
        orm.ready_at = asset.ready_at
        orm.deleted_at = asset.deleted_at
        orm.updated_at = asset.updated_at
        orm.version = asset.version


class SqlAlchemyVideoJobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, job: VideoJob) -> None:
        orm = VideoJobOrm(
            job_id=job.job_id,
            video_id=job.video_id,
            process_id=job.process_id,
            retry_of_job_id=job.retry_of_job_id,
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
            available_at=job.available_at,
            lease_owner=job.lease_owner,
            lease_token=job.lease_token,
            lease_expires_at=job.lease_expires_at,
            heartbeat_at=job.heartbeat_at,
            attempt_count=job.attempt_count,
            max_attempts=job.max_attempts,
            cancellation_requested=job.cancellation_requested,
            error_code=job.error_code,
            created_at=job.created_at,
            updated_at=job.updated_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            failed_at=job.failed_at,
            cancelled_at=job.cancelled_at,
            version=job.version,
            result_manifest_bucket=job.result_manifest_bucket,
            result_manifest_key=job.result_manifest_key,
            result_manifest_sha256=job.result_manifest_sha256,
            result_schema_version=job.result_schema_version,
        )
        self._session.add(orm)

    async def get_by_id(self, job_id: JobId) -> VideoJob | None:
        result = await self._session.execute(
            select(VideoJobOrm).where(VideoJobOrm.job_id == job_id)
        )
        orm = result.scalar_one_or_none()
        return _job_to_domain(orm) if orm else None

    async def get_by_process_id(self, process_id: ProcessId) -> VideoJob | None:
        result = await self._session.execute(
            select(VideoJobOrm).where(VideoJobOrm.process_id == process_id)
        )
        orm = result.scalar_one_or_none()
        return _job_to_domain(orm) if orm else None

    async def update(self, job: VideoJob) -> None:
        result = await self._session.execute(
            select(VideoJobOrm).where(VideoJobOrm.job_id == job.job_id)
        )
        orm = result.scalar_one_or_none()
        if orm is None:
            raise ValueError(f"VideoJob {job.job_id} not found")
        orm.state = job.state
        orm.stage = job.stage
        orm.progress_percent = job.progress_percent
        orm.processed_frames = job.processed_frames
        orm.sampled_frames = job.sampled_frames
        orm.detected_observations = job.detected_observations
        orm.person_count = job.person_count
        orm.lease_owner = job.lease_owner
        orm.lease_token = job.lease_token
        orm.lease_expires_at = job.lease_expires_at
        orm.heartbeat_at = job.heartbeat_at
        orm.attempt_count = job.attempt_count
        orm.cancellation_requested = job.cancellation_requested
        orm.error_code = job.error_code
        orm.started_at = job.started_at
        orm.completed_at = job.completed_at
        orm.failed_at = job.failed_at
        orm.cancelled_at = job.cancelled_at
        orm.result_manifest_bucket = job.result_manifest_bucket
        orm.result_manifest_key = job.result_manifest_key
        orm.result_manifest_sha256 = job.result_manifest_sha256
        orm.result_schema_version = job.result_schema_version
        orm.updated_at = job.updated_at
        orm.version = job.version


class SqlAlchemyIdempotencyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, scope: str, key_hash: str) -> dict[str, Any] | None:
        result = await self._session.execute(
            select(IdempotencyRecordOrm).where(
                IdempotencyRecordOrm.scope == scope,
                IdempotencyRecordOrm.key_hash == key_hash,
            )
        )
        orm = result.scalar_one_or_none()
        if orm is None:
            return None
        return {
            "scope": orm.scope,
            "key_hash": orm.key_hash,
            "request_hash": orm.request_hash,
            "state": orm.state,
            "resource_type": orm.resource_type,
            "resource_id": orm.resource_id,
            "response_status": orm.response_status,
            "response_snapshot": dict(orm.response_snapshot or {}),
            "expires_at": orm.expires_at,
        }

    async def upsert_in_progress(
        self,
        scope: str,
        key_hash: str,
        request_hash: str,
        expires_at: datetime,
    ) -> None:
        result = await self._session.execute(
            select(IdempotencyRecordOrm).where(
                IdempotencyRecordOrm.scope == scope,
                IdempotencyRecordOrm.key_hash == key_hash,
            )
        )
        orm = result.scalar_one_or_none()
        if orm is None:
            self._session.add(
                IdempotencyRecordOrm(
                    scope=scope,
                    key_hash=key_hash,
                    request_hash=request_hash,
                    state="in_progress",
                    expires_at=expires_at,
                )
            )
        else:
            orm.state = "in_progress"
            orm.request_hash = request_hash
            orm.expires_at = expires_at
            orm.updated_at = datetime.now(UTC)

    async def claim(
        self,
        scope: str,
        key_hash: str,
        request_hash: str,
        expires_at: datetime,
    ) -> dict[str, Any] | None:
        result = await self._session.execute(
            text(
                """
                INSERT INTO idempotency_record
                    (scope, key_hash, request_hash, state, expires_at)
                VALUES
                    (:scope, :key_hash, :request_hash, 'in_progress', :expires_at)
                ON CONFLICT (scope, key_hash) DO NOTHING
                """
            ),
            {
                "scope": scope,
                "key_hash": key_hash,
                "request_hash": request_hash,
                "expires_at": expires_at,
            },
        )
        if getattr(result, "rowcount", 0) == 1:
            return None
        return await self.get(scope, key_hash)

    async def set_in_progress(
        self,
        scope: str,
        key_hash: str,
        request_hash: str,
        expires_at: datetime,
    ) -> None:
        result = await self._session.execute(
            select(IdempotencyRecordOrm).where(
                IdempotencyRecordOrm.scope == scope,
                IdempotencyRecordOrm.key_hash == key_hash,
            )
        )
        orm = result.scalar_one_or_none()
        if orm is None:
            self._session.add(
                IdempotencyRecordOrm(
                    scope=scope,
                    key_hash=key_hash,
                    request_hash=request_hash,
                    state="in_progress",
                    expires_at=expires_at,
                )
            )
        else:
            orm.state = "in_progress"
            orm.request_hash = request_hash
            orm.expires_at = expires_at
            orm.updated_at = datetime.now(UTC)

    async def finalize(
        self,
        scope: str,
        key_hash: str,
        state: str,
        resource_type: str,
        resource_id: uuid.UUID,
        response_status: int,
        response_snapshot: dict[str, Any],
    ) -> None:
        result = await self._session.execute(
            select(IdempotencyRecordOrm).where(
                IdempotencyRecordOrm.scope == scope,
                IdempotencyRecordOrm.key_hash == key_hash,
            )
        )
        orm = result.scalar_one_or_none()
        if orm is None:
            raise ValueError(f"Idempotency record {scope}/{key_hash} not found")
        orm.state = state
        orm.resource_type = resource_type
        orm.resource_id = resource_id
        orm.response_status = response_status
        orm.response_snapshot = response_snapshot
        orm.completed_at = datetime.now(UTC)
        orm.updated_at = datetime.now(UTC)

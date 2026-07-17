"""End-to-end video processing orchestration for a claimed job."""

from __future__ import annotations

import json
import logging
import math
import uuid
from datetime import UTC, datetime
from typing import Any, Protocol

from app.application.ports.object_store import ObjectStore
from app.application.ports.track_crop_provider import TrackCropProvider
from app.application.ports.unit_of_work import UnitOfWorkFactory
from app.application.ports.video_observations import VideoObservationFrame
from app.application.services.identity_storage_lifecycle_service import (
    IdentityStorageLifecycleService,
)
from app.application.services.video_identity_resolution_service import (
    TrackIdentityOutcome,
    VideoIdentityResolutionService,
)
from app.application.services.video_overlay_service import VideoOverlayService
from app.application.services.video_reconciliation_service import VideoReconciliationService
from app.application.services.video_track_persistence_service import (
    VideoTrackPersistenceService,
)
from app.application.services.video_tracking_service import VideoTrackingService
from app.domain.entities.video_job import VideoJob
from app.domain.entities.video_timeline_chunk import VideoTimelineChunk
from app.domain.entities.video_tracking import CanonicalTrack
from app.domain.errors import JobNotFoundError
from app.domain.value_objects import JobId, ProcessId

logger = logging.getLogger(__name__)


class _VideoJobRepository(Protocol):
    async def get_by_id(self, job_id: JobId) -> VideoJob | None: ...

    async def update(self, job: VideoJob) -> None: ...


class VideoProcessingService:
    def __init__(
        self,
        *,
        unit_of_work_factory: UnitOfWorkFactory,
        object_store: ObjectStore,
        lifecycle_service: IdentityStorageLifecycleService,
        tracking_service: VideoTrackingService,
        reconciliation_service: VideoReconciliationService,
        identity_resolution_service: VideoIdentityResolutionService,
        track_persistence_service: VideoTrackPersistenceService,
        overlay_service: VideoOverlayService,
        crop_provider: TrackCropProvider,
        bucket_name: str,
        result_prefix: str,
        manifest_schema_version: str = "1",
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._object_store = object_store
        self._lifecycle = lifecycle_service
        self._tracking = tracking_service
        self._reconciliation = reconciliation_service
        self._identity = identity_resolution_service
        self._persistence = track_persistence_service
        self._overlay = overlay_service
        self._crop_provider = crop_provider
        self._bucket_name = bucket_name
        self._result_prefix = result_prefix.rstrip("/") + "/"
        self._manifest_schema_version = manifest_schema_version

    async def process(
        self,
        job_id: JobId,
        frames: list[VideoObservationFrame],
    ) -> None:
        async with self._unit_of_work_factory() as uow:
            job = await uow.video_jobs.get_by_id(job_id)
            if job is None:
                raise JobNotFoundError(f"Job {job_id} not found")

            if job.cancellation_requested and job.state in ("processing", "cancelling"):
                job.state = "cancelled"
                job.stage = "cleanup"
                job.cancelled_at = datetime.now(UTC)
                self._clear_lease(job)
                await uow.video_jobs.update(job)
                await uow.commit()
                return

            job.stage = "track_reconcile"
            await uow.video_jobs.update(job)
            await uow.commit()

        tracklets = self._tracking.build_raw_tracklets(frames, job_id=job.job_id)
        canonical_tracks = self._reconciliation.reconcile_tracklets(tracklets)
        crop_bytes_by_track_id = {
            track.track_id: await self._crop_provider.get_crop(track.track_id, track=track)
            for track in canonical_tracks
        }
        outcomes = await self._identity.resolve(
            process_id=job.process_id,
            canonical_tracks=canonical_tracks,
            crop_bytes_by_track_id=crop_bytes_by_track_id,
        )
        track_outcomes: list[tuple[CanonicalTrack, TrackIdentityOutcome]] = list(
            zip(canonical_tracks, outcomes, strict=True)
        )

        await self._persistence.persist_tracks(job_id, track_outcomes)

        overlay_chunk = await self._overlay.write_public_overlay(
            job_id=job_id,
            video_id=job.video_id,
            canonical_tracks=canonical_tracks,
            outcomes=outcomes,
        )

        manifest_key = self._manifest_key(job.video_id, job_id)
        manifest_data = self._build_manifest(job, track_outcomes, overlay_chunk)
        manifest_json = json.dumps(manifest_data).encode("utf-8")
        stat = await self._object_store.upload(
            manifest_key, manifest_json, "application/json"
        )

        async with self._unit_of_work_factory() as uow:
            job = await uow.video_jobs.get_by_id(job_id)
            if job is None:
                raise JobNotFoundError(f"Job {job_id} disappeared during processing")
            job.state = "completed"
            job.stage = "finalize"
            job.progress_percent = 100
            job.sampled_frames = len(frames)
            job.processed_frames = len(frames)
            job.detected_observations = sum(
                len(tracklet.detections) for track in canonical_tracks for tracklet in track.tracklets
            )
            job.person_count = len(canonical_tracks)
            job.result_manifest_bucket = stat.bucket
            job.result_manifest_key = stat.key
            job.result_manifest_sha256 = stat.sha256
            job.result_schema_version = self._manifest_schema_version
            job.completed_at = datetime.now(UTC)
            self._clear_lease(job)
            await uow.video_jobs.update(job)

            await self._lifecycle.complete_process(
                process_id=ProcessId(job.process_id),
                face_count=len(canonical_tracks),
                details={"job_id": str(job_id), "video_id": str(job.video_id)},
            )
            await uow.commit()

    @staticmethod
    def _clear_lease(job: VideoJob) -> None:
        job.lease_owner = None
        job.lease_token = None
        job.lease_expires_at = None
        job.heartbeat_at = None

    def _manifest_key(self, video_id: uuid.UUID, job_id: JobId) -> str:
        return f"{self._result_prefix}{video_id}/jobs/{job_id}/result/manifest.json"

    def _build_manifest(
        self,
        job: VideoJob,
        track_outcomes: list[tuple[CanonicalTrack, TrackIdentityOutcome]],
        overlay_chunk: VideoTimelineChunk,
    ) -> dict[str, Any]:
        people = [
            {
                "trackId": str(track.track_id),
                "faceId": str(outcome.face_id),
                "status": outcome.status,
                "matchConfidence": math.nan if outcome.match_confidence is None else outcome.match_confidence,
                "firstFrameIndex": track.appearance_intervals[0].start_frame_index
                if track.appearance_intervals
                else 0,
                "lastFrameIndex": track.appearance_intervals[-1].end_frame_index
                if track.appearance_intervals
                else 0,
            }
            for track, outcome in track_outcomes
        ]
        return {
            "schemaVersion": self._manifest_schema_version,
            "jobId": str(job.job_id),
            "videoId": str(job.video_id),
            "processId": str(job.process_id),
            "generatedAt": datetime.now(UTC).isoformat(),
            "personCount": len(track_outcomes),
            "people": people,
            "overlay": {
                "bucket": overlay_chunk.bucket,
                "objectKey": overlay_chunk.object_key,
                "sha256": overlay_chunk.content_sha256,
                "recordCount": overlay_chunk.record_count,
            },
        }

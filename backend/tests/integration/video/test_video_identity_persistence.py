"""M7 integration tests: canonical video track persistence to PostgreSQL."""

from __future__ import annotations

from app.application.ports.unit_of_work import UnitOfWork, UnitOfWorkFactory
from app.application.services.video_identity_resolution_service import TrackIdentityOutcome
from app.application.services.video_reconciliation_service import VideoReconciliationService
from app.application.services.video_track_persistence_service import VideoTrackPersistenceService
from app.application.services.video_tracking_service import VideoTrackingService
from app.domain.entities.face_identity import FaceIdentity
from app.domain.entities.process_record import ProcessRecord
from app.domain.entities.recognition_result import RecognitionResult
from app.domain.entities.video_asset import VideoAsset
from app.domain.entities.video_job import VideoJob
from app.domain.entities.video_tracking import CanonicalTrack
from app.domain.value_objects import (
    BoundingBox,
    FaceId,
    JobId,
    ProcessId,
    ResultId,
    UploadSessionId,
    VideoId,
)
from app.infrastructure.persistence.sqlalchemy.session import async_session_maker
from app.infrastructure.persistence.sqlalchemy.unit_of_work import SqlAlchemyUnitOfWork
from app.infrastructure.uuid7 import generate_uuid7
from tests.fixtures.embedding_fixtures import DIMENSION


def _unit_vector(first_one_at: int) -> tuple[float, ...]:
    vector = [0.0] * DIMENSION
    vector[first_one_at] = 1.0
    return tuple(vector)


VECTOR_A = _unit_vector(0)


def _factory() -> UnitOfWork:
    return SqlAlchemyUnitOfWork(async_session_maker)


def _build_canonical_track(embedding: tuple[float, ...], frames: list[int]) -> CanonicalTrack:
    from app.application.ports.video_observations import FaceObservation, VideoObservationFrame

    def det(frame_index: int) -> FaceObservation:
        return FaceObservation(
            detection_id=f"d-{frame_index}",
            ordinal=0,
            bbox=BoundingBox(x=10, y=0, width=20, height=20),
            landmarks=(0.0,) * 10,
            detector_score=0.9,
            quality_score=0.9,
            tracking_eligible=True,
            recognition_eligible=True,
            rejection_code="",
            embedding=embedding,
            model_version="retinaface_r50_glintr100_v1",
            preprocess_version="cuda_nv12_align_v1",
        )

    observations = [
        VideoObservationFrame(
            job_id="job",
            video_id="video",
            stream_index=0,
            frame_index=f,
            source_pts=f,
            pts_ns=f * 33_000_000,
            display_width=640,
            display_height=480,
            detections=(det(f),),
        )
        for f in frames
    ]
    tracker = VideoTrackingService(max_gap_frames=1, iou_threshold=0.3)
    tracklets = tracker.build_raw_tracklets(observations)
    reconciler = VideoReconciliationService(merge_threshold=0.9)
    return reconciler.reconcile_tracklets(tracklets)[0]


async def test_persist_video_track_to_postgresql() -> None:
    factory: UnitOfWorkFactory = _factory
    service = VideoTrackPersistenceService(factory)
    job_id = JobId(generate_uuid7())
    process_id = ProcessId(generate_uuid7())
    video_id = VideoId(generate_uuid7())
    face_id = FaceId(generate_uuid7())
    result_id = ResultId(generate_uuid7())
    track = _build_canonical_track(VECTOR_A, [0, 1, 2])

    async with factory() as uow:
        asset = VideoAsset(
            video_id=video_id,
            upload_session_id=UploadSessionId(generate_uuid7()),
            state="uploading",
        )
        await uow.video_assets.add(asset)
        job = VideoJob(job_id=job_id, video_id=video_id, process_id=process_id)
        await uow.video_jobs.add(job)
        identity = FaceIdentity(face_id=face_id, status="anonymous")
        await uow.face_identities.add(identity)
        process = ProcessRecord(
            process_id=process_id,
            process_type="video_recognize",
        )
        await uow.processes.add(process)
        result = RecognitionResult(
            result_id=result_id,
            process_id=process_id,
            face_id=face_id,
            status="new_anonymous",
            bounding_box=BoundingBox(x=0, y=0, width=10, height=10),
            match_confidence=0.0,
        )
        await uow.recognition_results.add(result)
        await uow.commit()

    outcome = TrackIdentityOutcome(
        track_id=track.track_id,
        face_id=face_id,
        sample_id=None,
        result_id=result_id,
        status="new_anonymous",
        match_confidence=0.0,
        top1_score=None,
        top2_score=None,
        margin_score=None,
        threshold_used=0.95,
    )

    await service.persist_tracks(job_id=job_id, track_outcomes=[(track, outcome)])

    async with factory() as uow:
        tracks = await uow.video_tracks.list_by_job_id(job_id)
        assert len(tracks) == 1
        persisted = tracks[0]
        assert persisted.track_id == track.track_id
        assert persisted.face_id == face_id
        assert persisted.recognition_result_id == result_id
        assert persisted.status_at_processing == "new_anonymous"

        tracklets = await uow.video_tracklets.list_by_track_id(track.track_id)
        assert len(tracklets) == 1

        intervals = await uow.video_appearance_intervals.list_by_track_id(track.track_id)
        assert len(intervals) == len(track.appearance_intervals)

        links = await uow.video_track_samples.list_by_track_id(track.track_id)
        assert len(links) == 0

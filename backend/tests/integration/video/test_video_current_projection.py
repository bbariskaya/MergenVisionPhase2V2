"""M3 integration test: video result reflects current identity projection.

Historical track snapshot must stay immutable while the read-side people
endpoint exposes the latest face_identity status/name/metadata.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.application.ports.video_observations import FaceObservation, VideoObservationFrame
from app.application.services.identity_storage_lifecycle_service import (
    IdentityStorageLifecycleService,
)
from app.application.services.video_identity_resolution_service import (
    VideoIdentityResolutionService,
)
from app.application.services.video_overlay_service import VideoOverlayService
from app.application.services.video_processing_service import VideoProcessingService
from app.application.services.video_reconciliation_service import VideoReconciliationService
from app.application.services.video_track_persistence_service import (
    VideoTrackPersistenceService,
)
from app.application.services.image_recognition_service import ImageRecognitionService
from app.application.services.video_tracking_service import VideoTrackingService
from app.domain.entities.process_record import ProcessRecord
from app.domain.entities.video_asset import VideoAsset
from app.domain.entities.video_job import VideoJob
from app.domain.value_objects import BoundingBox, JobId, ProcessId, VideoId
from app.infrastructure.persistence.sqlalchemy.session import async_session_maker
from app.infrastructure.persistence.sqlalchemy.unit_of_work import SqlAlchemyUnitOfWork
from app.infrastructure.runtime.track_crop_provider import PlaceholderTrackCropProvider
from app.infrastructure.storage.minio_adapter import MinIOObjectStore
from app.infrastructure.uuid7 import Uuid7Generator, generate_uuid7
from app.infrastructure.vectors.qdrant_adapter import QdrantVectorStore


def _unit_vector(seed: int) -> tuple[float, ...]:
    vec = [0.0] * 512
    vec[seed % 512] = 1.0
    return tuple(vec)


def _embedding_for_job(job_id: UUID) -> tuple[float, ...]:
    digest = hashlib.sha256(str(job_id).encode()).digest()
    seed = int.from_bytes(digest[:8], "big")
    return _unit_vector(seed)


def _observation_frames(video_id: UUID, job_id: UUID) -> list[VideoObservationFrame]:
    embedding = _embedding_for_job(job_id)
    frames: list[VideoObservationFrame] = []
    for frame_index in range(5):
        frames.append(
            VideoObservationFrame(
                job_id=str(job_id),
                video_id=str(video_id),
                stream_index=0,
                frame_index=frame_index,
                source_pts=frame_index * 3000,
                pts_ns=frame_index * 33_000_000,
                display_width=640,
                display_height=480,
                detections=(
                    FaceObservation(
                        detection_id=f"det-{frame_index}-0",
                        ordinal=0,
                        bbox=BoundingBox(x=100 + frame_index, y=100, width=80, height=90),
                        landmarks=tuple(10.0 for _ in range(10)),
                        detector_score=0.95,
                        quality_score=0.9,
                        tracking_eligible=True,
                        recognition_eligible=True,
                        rejection_code="",
                        embedding=embedding,
                        model_version="retinaface_r50_glintr100_v1",
                        preprocess_version="cuda_five_point_align",
                    ),
                ),
            )
        )
    return frames


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def test_bucket() -> str:
    from app.infrastructure.config import settings

    return settings.minio_bucket_name


@pytest.mark.asyncio
async def test_video_people_reflects_current_identity_projection(
    client: TestClient,
    test_bucket: str,
) -> None:
    object_store = MinIOObjectStore()
    vector_store = QdrantVectorStore()
    id_generator = Uuid7Generator()
    uow_factory = lambda: SqlAlchemyUnitOfWork(async_session_maker)  # noqa: E731

    lifecycle = IdentityStorageLifecycleService(
        unit_of_work_factory=uow_factory,
        object_store=object_store,
        vector_store=vector_store,
        id_generator=id_generator,
    )
    tracking_service = VideoTrackingService(max_gap_frames=2, iou_threshold=0.3)
    reconciliation_service = VideoReconciliationService(merge_threshold=0.6)
    identity_service = VideoIdentityResolutionService(
        lifecycle=lifecycle,
        match_threshold=0.55,
        margin_multiplier=0.95,
    )
    persistence_service = VideoTrackPersistenceService(unit_of_work_factory=uow_factory)
    overlay_service = VideoOverlayService(
        unit_of_work_factory=uow_factory,
        object_store=object_store,
        bucket_name=test_bucket,
    )
    crop_provider = PlaceholderTrackCropProvider(b"fake-crop-for-test")

    service = VideoProcessingService(
        unit_of_work_factory=uow_factory,
        object_store=object_store,
        lifecycle_service=lifecycle,
        tracking_service=tracking_service,
        reconciliation_service=reconciliation_service,
        identity_resolution_service=identity_service,
        track_persistence_service=persistence_service,
        overlay_service=overlay_service,
        crop_provider=crop_provider,
        bucket_name=test_bucket,
        result_prefix="videos/",
    )

    video_id = VideoId(generate_uuid7())
    process_id = ProcessId(generate_uuid7())
    job_id = JobId(generate_uuid7())

    async with uow_factory() as uow:
        process = ProcessRecord(
            process_id=process_id,
            process_type="video_recognize",
            status="processing",
            details={"request_id": "r1", "video_id": str(video_id), "job_id": str(job_id)},
        )
        await uow.processes.add(process)

        asset = VideoAsset(
            video_id=video_id,
            upload_session_id=generate_uuid7(),
            state="ready",
            bucket=test_bucket,
            object_key=f"videos/{video_id}/source/original",
            content_sha256="a" * 64,
            size_bytes=1024,
            content_type="video/mp4",
            container_format="mp4",
            video_codec="h264",
            display_width=640,
            display_height=480,
            duration_ns=165_000_000,
            time_base_num=1,
            time_base_den=30_000,
            total_frames=5,
            retention_until=datetime.now(UTC),
            ready_at=datetime.now(UTC),
        )
        await uow.video_assets.add(asset)

        job = VideoJob(
            job_id=job_id,
            video_id=video_id,
            process_id=process_id,
            state="processing",
            stage="decode_infer",
            lease_owner="test-worker",
            lease_token=generate_uuid7(),
            lease_expires_at=datetime.now(UTC),
            heartbeat_at=datetime.now(UTC),
        )
        await uow.video_jobs.add(job)
        await uow.commit()

    await object_store.upload(
        f"videos/{video_id}/source/original",
        b"fake-video-bytes",
        "video/mp4",
    )

    frames = _observation_frames(video_id, job_id)
    await service.process(job_id, frames)

    async with uow_factory() as uow:
        tracks = await uow.video_tracks.list_by_job_id(job_id)
        assert len(tracks) == 1
        face_id = tracks[0].face_id
        assert tracks[0].status_at_processing == "new_anonymous"
        assert tracks[0].name_at_processing is None

    people_response = client.get(f"/api/v1/videos/jobs/{job_id}/people")
    assert people_response.status_code == 200, people_response.text
    person = people_response.json()["people"][0]
    assert person["faceId"] == str(face_id)
    assert person["status"] == "new_anonymous"
    assert person["name"] is None
    assert person["currentStatus"] == "anonymous"
    assert person["currentName"] is None

    await lifecycle.enroll_identity(face_id, "Alice", {"department": "IT"})

    people_response = client.get(f"/api/v1/videos/jobs/{job_id}/people")
    assert people_response.status_code == 200, people_response.text
    person = people_response.json()["people"][0]

    assert person["status"] == "new_anonymous"
    assert person["name"] is None
    assert person["currentStatus"] == "known"
    assert person["currentName"] == "Alice"
    assert person["currentMetadata"] == {"department": "IT"}

    service_update = ImageRecognitionService(
        lifecycle_service=lifecycle,
        unit_of_work_factory=uow_factory,
        max_image_bytes=10_000_000,
        model_version="retinaface_r50_glintr100_v1",
        engine=None,  # type: ignore[arg-type]
        engine_factory=lambda: None,  # type: ignore[return-value]
        match_threshold=0.55,
    )
    updated = await service_update.update_identity(face_id, "Alice Smith", {"department": "HR"})
    assert updated.display_name == "Alice Smith"

    people_response = client.get(f"/api/v1/videos/jobs/{job_id}/people")
    assert people_response.status_code == 200, people_response.text
    person = people_response.json()["people"][0]

    assert person["status"] == "new_anonymous"
    assert person["name"] is None
    assert person["currentStatus"] == "known"
    assert person["currentName"] == "Alice Smith"
    assert person["currentMetadata"] == {"department": "HR"}

    async with uow_factory() as uow:
        track = await uow.video_tracks.get_by_id(list(tracks)[0].track_id)
        assert track is not None
        assert track.status_at_processing == "new_anonymous"
        assert track.name_at_processing is None

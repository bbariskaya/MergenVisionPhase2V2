from __future__ import annotations

from app.application.ports.track_crop_provider import TrackCropProvider
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
from app.application.services.video_tracking_service import VideoTrackingService
from app.infrastructure.config import settings
from app.infrastructure.persistence.sqlalchemy.session import async_session_maker
from app.infrastructure.persistence.sqlalchemy.unit_of_work import SqlAlchemyUnitOfWork
from app.infrastructure.storage.minio_adapter import MinIOObjectStore
from app.infrastructure.uuid7 import Uuid7Generator
from app.infrastructure.vectors.qdrant_adapter import QdrantVectorStore


def build_video_processing_service(crop_provider: TrackCropProvider) -> VideoProcessingService:
    def unit_of_work_factory() -> SqlAlchemyUnitOfWork:
        return SqlAlchemyUnitOfWork(async_session_maker)

    object_store = MinIOObjectStore()
    vector_store = QdrantVectorStore()
    id_generator = Uuid7Generator()

    lifecycle = IdentityStorageLifecycleService(
        unit_of_work_factory=unit_of_work_factory,
        object_store=object_store,
        vector_store=vector_store,
        id_generator=id_generator,
    )

    return VideoProcessingService(
        unit_of_work_factory=unit_of_work_factory,
        object_store=object_store,
        lifecycle_service=lifecycle,
        tracking_service=VideoTrackingService(
            max_gap_frames=2,
            iou_threshold=0.3,
            id_generator=id_generator.new_uuid7,
        ),
        reconciliation_service=VideoReconciliationService(
            merge_threshold=0.6,
            id_generator=id_generator.new_uuid7,
        ),
        identity_resolution_service=VideoIdentityResolutionService(
            lifecycle=lifecycle,
            match_threshold=settings.match_threshold,
        ),
        track_persistence_service=VideoTrackPersistenceService(
            unit_of_work_factory=unit_of_work_factory,
            id_generator=id_generator.new_uuid7,
        ),
        overlay_service=VideoOverlayService(
            unit_of_work_factory=unit_of_work_factory,
            object_store=object_store,
            bucket_name=settings.minio_bucket_name,
            id_generator=id_generator.new_uuid7,
        ),
        crop_provider=crop_provider,
        bucket_name=settings.minio_bucket_name,
        result_prefix="videos/",
    )

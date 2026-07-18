"""SQLAlchemy ORM models."""

from app.infrastructure.persistence.sqlalchemy.models.appearance_interval import (
    AppearanceIntervalOrm,
)
from app.infrastructure.persistence.sqlalchemy.models.face_identity import FaceIdentityOrm
from app.infrastructure.persistence.sqlalchemy.models.face_sample import FaceSampleOrm
from app.infrastructure.persistence.sqlalchemy.models.idempotency_record import IdempotencyRecordOrm
from app.infrastructure.persistence.sqlalchemy.models.outbox_event import OutboxEventOrm
from app.infrastructure.persistence.sqlalchemy.models.person import PersonOrm
from app.infrastructure.persistence.sqlalchemy.models.process_event import ProcessEventOrm
from app.infrastructure.persistence.sqlalchemy.models.process_record import ProcessRecordOrm
from app.infrastructure.persistence.sqlalchemy.models.recognition_result import RecognitionResultOrm
from app.infrastructure.persistence.sqlalchemy.models.video_asset import VideoAssetOrm
from app.infrastructure.persistence.sqlalchemy.models.video_job import VideoJobOrm
from app.infrastructure.persistence.sqlalchemy.models.video_timeline_chunk import (
    VideoTimelineChunkOrm,
)
from app.infrastructure.persistence.sqlalchemy.models.video_track import VideoTrackOrm
from app.infrastructure.persistence.sqlalchemy.models.video_track_sample import VideoTrackSampleOrm
from app.infrastructure.persistence.sqlalchemy.models.video_tracklet import VideoTrackletOrm

__all__ = [
    "AppearanceIntervalOrm",
    "FaceIdentityOrm",
    "FaceSampleOrm",
    "IdempotencyRecordOrm",
    "PersonOrm",
    "OutboxEventOrm",
    "ProcessEventOrm",
    "ProcessRecordOrm",
    "RecognitionResultOrm",
    "VideoAssetOrm",
    "VideoJobOrm",
    "VideoTimelineChunkOrm",
    "VideoTrackOrm",
    "VideoTrackSampleOrm",
    "VideoTrackletOrm",
]

"""Repository ports."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.domain.entities.face_identity import FaceIdentity
from app.domain.entities.face_sample import FaceSample
from app.domain.entities.process_record import ProcessRecord
from app.domain.entities.recognition_result import RecognitionResult
from app.domain.entities.video_timeline_chunk import VideoTimelineChunk
from app.domain.entities.video_track import (
    VideoAppearanceInterval,
    VideoTrack,
    VideoTracklet,
    VideoTrackSample,
)
from app.domain.value_objects import FaceId, JobId, ProcessId, SampleId


class FaceIdentityRepository(ABC):
    @abstractmethod
    async def add(self, identity: FaceIdentity) -> None: ...

    @abstractmethod
    async def get_by_id(self, face_id: FaceId) -> FaceIdentity | None: ...

    @abstractmethod
    async def get_active_by_id(self, face_id: FaceId) -> FaceIdentity | None: ...

    @abstractmethod
    async def update(self, identity: FaceIdentity) -> None: ...

    @abstractmethod
    async def update_with_expected_version(
        self,
        identity: FaceIdentity,
        expected_version: int,
    ) -> FaceIdentity: ...

    @abstractmethod
    async def list_all(self) -> Sequence[FaceIdentity]: ...

    @abstractmethod
    async def search(
        self,
        query: str | None = None,
        status: str | None = None,
        is_active: bool = True,
    ) -> Sequence[FaceIdentity]: ...


class FaceSampleRepository(ABC):
    @abstractmethod
    async def add(self, sample: FaceSample) -> None: ...

    @abstractmethod
    async def get_by_id(self, sample_id: SampleId) -> FaceSample | None: ...

    @abstractmethod
    async def list_active_by_face_id(self, face_id: FaceId) -> Sequence[FaceSample]: ...

    @abstractmethod
    async def update(self, sample: FaceSample) -> None: ...


class ProcessRepository(ABC):
    @abstractmethod
    async def add(self, process: ProcessRecord) -> None: ...

    @abstractmethod
    async def get_by_id(self, process_id: ProcessId) -> ProcessRecord | None: ...

    @abstractmethod
    async def update(self, process: ProcessRecord) -> None: ...

    @abstractmethod
    async def list_by_status(self, status: str) -> Sequence[ProcessRecord]: ...


class RecognitionResultRepository(ABC):
    @abstractmethod
    async def add(self, result: RecognitionResult) -> None: ...

    @abstractmethod
    async def list_by_process_id(self, process_id: ProcessId) -> Sequence[RecognitionResult]: ...

    @abstractmethod
    async def list_by_face_id(
        self,
        face_id: FaceId,
        limit: int | None = None,
    ) -> Sequence[RecognitionResult]: ...


class VideoTrackRepository(ABC):
    @abstractmethod
    async def add(self, track: VideoTrack) -> None: ...

    @abstractmethod
    async def get_by_id(self, track_id: uuid.UUID) -> VideoTrack | None: ...

    @abstractmethod
    async def list_by_job_id(self, job_id: JobId) -> Sequence[VideoTrack]: ...

    @abstractmethod
    async def update(self, track: VideoTrack) -> None: ...

    @abstractmethod
    async def delete_by_job_id(self, job_id: JobId) -> int: ...


class VideoTrackletRepository(ABC):
    @abstractmethod
    async def add(self, tracklet: VideoTracklet) -> None: ...

    @abstractmethod
    async def list_by_track_id(self, track_id: uuid.UUID) -> Sequence[VideoTracklet]: ...

    @abstractmethod
    async def list_by_job_id(self, job_id: JobId) -> Sequence[VideoTracklet]: ...

    @abstractmethod
    async def delete_by_job_id(self, job_id: JobId) -> int: ...


class VideoAppearanceIntervalRepository(ABC):
    @abstractmethod
    async def add(self, interval: VideoAppearanceInterval) -> None: ...

    @abstractmethod
    async def list_by_track_id(self, track_id: uuid.UUID) -> Sequence[VideoAppearanceInterval]: ...

    @abstractmethod
    async def delete_by_job_id(self, job_id: JobId) -> int: ...


class VideoTrackSampleRepository(ABC):
    @abstractmethod
    async def add(self, link: VideoTrackSample) -> None: ...

    @abstractmethod
    async def list_by_track_id(self, track_id: uuid.UUID) -> Sequence[VideoTrackSample]: ...

    @abstractmethod
    async def delete_by_job_id(self, job_id: JobId) -> int: ...


class VideoTimelineChunkRepository(ABC):
    @abstractmethod
    async def add(self, chunk: VideoTimelineChunk) -> None: ...

    @abstractmethod
    async def delete_by_job_id(self, job_id: JobId) -> int: ...

    @abstractmethod
    async def list_by_job_id(
        self,
        job_id: JobId,
        artifact_kind: str | None = None,
    ) -> Sequence[VideoTimelineChunk]: ...

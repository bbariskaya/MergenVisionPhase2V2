"""M7 video track persistence service unit tests."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Self

from app.application.ports.repositories import (
    FaceIdentityRepository,
    VideoAppearanceIntervalRepository,
    VideoTimelineChunkRepository,
    VideoTrackletRepository,
    VideoTrackRepository,
    VideoTrackSampleRepository,
)
from app.application.ports.unit_of_work import UnitOfWork
from app.application.services.video_identity_resolution_service import TrackIdentityOutcome
from app.application.services.video_track_persistence_service import VideoTrackPersistenceService
from app.domain.entities.video_track import (
    VideoAppearanceInterval,
    VideoTrack,
    VideoTracklet,
    VideoTrackSample,
)
from app.domain.value_objects import FaceId, JobId, ResultId, SampleId
from app.infrastructure.uuid7 import generate_uuid7
from tests.unit.services.test_video_identity_resolution_service import (
    VECTOR_A,
    _build_canonical_track,
)


@dataclass
class _FakeTrackRepo(VideoTrackRepository):
    tracks: list[VideoTrack] = field(default_factory=list)

    async def add(self, track: VideoTrack) -> None:
        self.tracks.append(track)

    async def get_by_id(self, track_id: uuid.UUID) -> VideoTrack | None:
        for t in self.tracks:
            if t.track_id == track_id:
                return t
        return None

    async def list_by_job_id(self, job_id: JobId) -> Any:
        return [t for t in self.tracks if t.job_id == job_id]

    async def update(self, track: VideoTrack) -> None:
        for i, t in enumerate(self.tracks):
            if t.track_id == track.track_id:
                self.tracks[i] = track
                return

    async def delete_by_job_id(self, job_id: JobId) -> int:
        before = len(self.tracks)
        self.tracks = [t for t in self.tracks if t.job_id != job_id]
        return before - len(self.tracks)


@dataclass
class _FakeTrackletRepo(VideoTrackletRepository):
    tracklets: list[VideoTracklet] = field(default_factory=list)

    async def add(self, tracklet: VideoTracklet) -> None:
        self.tracklets.append(tracklet)

    async def list_by_track_id(self, track_id: uuid.UUID) -> Any:
        return [tl for tl in self.tracklets if tl.track_id == track_id]

    async def list_by_job_id(self, job_id: JobId) -> Any:
        return [tl for tl in self.tracklets if tl.job_id == job_id]

    async def delete_by_job_id(self, job_id: JobId) -> int:
        before = len(self.tracklets)
        self.tracklets = [tl for tl in self.tracklets if tl.job_id != job_id]
        return before - len(self.tracklets)


@dataclass
class _FakeIntervalRepo(VideoAppearanceIntervalRepository):
    intervals: list[VideoAppearanceInterval] = field(default_factory=list)

    async def add(self, interval: VideoAppearanceInterval) -> None:
        self.intervals.append(interval)

    async def list_by_track_id(self, track_id: uuid.UUID) -> Any:
        return [iv for iv in self.intervals if iv.track_id == track_id]

    async def delete_by_job_id(self, job_id: JobId) -> int:
        before = len(self.intervals)
        self.intervals = [iv for iv in self.intervals if iv.job_id != job_id]
        return before - len(self.intervals)


@dataclass
class _FakeSampleRepo(VideoTrackSampleRepository):
    samples: list[VideoTrackSample] = field(default_factory=list)

    async def add(self, link: VideoTrackSample) -> None:
        self.samples.append(link)

    async def list_by_track_id(self, track_id: uuid.UUID) -> Any:
        return [s for s in self.samples if s.track_id == track_id]

    async def delete_by_job_id(self, job_id: JobId) -> int:
        before = len(self.samples)
        self.samples = [s for s in self.samples if s.track_id not in {
            t.track_id for t in getattr(job_id, "_tracks", [])
        }]
        return before - len(self.samples)


@dataclass
class _FakeTimelineChunkRepo(VideoTimelineChunkRepository):
    chunks: list[Any] = field(default_factory=list)

    async def add(self, chunk: Any) -> None:
        self.chunks.append(chunk)

    async def list_by_job_id(
        self,
        job_id: JobId,
        artifact_kind: str | None = None,
    ) -> Any:
        return self.chunks

    async def delete_by_job_id(self, job_id: JobId) -> int:
        before = len(self.chunks)
        self.chunks = [c for c in self.chunks if getattr(c, "job_id", None) != job_id]
        return before - len(self.chunks)


@dataclass
class _FakeFaceIdentityRepo(FaceIdentityRepository):
    async def get_active_by_id(self, face_id: FaceId) -> Any:
        return None

    async def get_by_id(self, face_id: FaceId) -> Any:
        return None

    async def list_all(self) -> Any:
        return []

    async def search(
        self,
        query: str | None = None,
        status: str | None = None,
        is_active: bool = True,
    ) -> Any:
        return []

    async def add(self, identity: Any) -> None:
        pass

    async def update(self, identity: Any) -> None:
        pass

    async def update_with_expected_version(self, identity: Any, expected_version: int) -> Any:
        pass


@dataclass
class _FakeUow(UnitOfWork):
    face_identities: _FakeFaceIdentityRepo = field(default_factory=_FakeFaceIdentityRepo)
    video_tracks: _FakeTrackRepo = field(default_factory=_FakeTrackRepo)
    video_tracklets: _FakeTrackletRepo = field(default_factory=_FakeTrackletRepo)
    video_appearance_intervals: _FakeIntervalRepo = field(default_factory=_FakeIntervalRepo)
    video_track_samples: _FakeSampleRepo = field(default_factory=_FakeSampleRepo)
    video_timeline_chunks: _FakeTimelineChunkRepo = field(default_factory=_FakeTimelineChunkRepo)
    committed: int = 0
    rolled_back: bool = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_val is not None:
            self.rolled_back = True

    async def commit(self) -> None:
        self.committed += 1

    async def rollback(self) -> None:
        self.rolled_back = True

    async def flush(self) -> None:
        pass


@dataclass
class _FakeUowFactory:
    uow: _FakeUow = field(default_factory=_FakeUow)

    def __call__(self) -> UnitOfWork:
        return self.uow


def _outcome_for_track(track_id: uuid.UUID, sample_id: SampleId) -> TrackIdentityOutcome:
    return TrackIdentityOutcome(
        track_id=track_id,
        face_id=FaceId(generate_uuid7()),
        sample_id=sample_id,
        result_id=ResultId(generate_uuid7()),
        status="new_anonymous",
        match_confidence=0.0,
        top1_score=None,
        top2_score=None,
        margin_score=None,
        threshold_used=0.95,
    )


async def test_persists_track_tracklets_intervals_and_sample_link() -> None:
    factory = _FakeUowFactory()
    service = VideoTrackPersistenceService(factory)
    job_id = JobId(generate_uuid7())
    track = _build_canonical_track(VECTOR_A, [0, 1, 2])
    sample_id = SampleId(generate_uuid7())
    outcome = _outcome_for_track(track.track_id, sample_id)

    persisted = await service.persist_tracks(
        job_id=job_id,
        track_outcomes=[(track, outcome)],
    )

    uow = factory.uow
    assert uow.committed == 1
    assert not uow.rolled_back
    assert len(persisted) == 1
    assert len(uow.video_tracks.tracks) == 1
    persisted_track = uow.video_tracks.tracks[0]
    assert persisted_track.track_id == track.track_id
    assert persisted_track.job_id == job_id
    assert persisted_track.face_id == outcome.face_id
    assert persisted_track.status_at_processing == "new_anonymous"
    assert persisted_track.recognition_result_id == outcome.result_id
    assert persisted_track.best_sample_id == sample_id

    assert len(uow.video_tracklets.tracklets) == 1
    assert uow.video_tracklets.tracklets[0].track_id == track.track_id

    assert len(uow.video_appearance_intervals.intervals) == len(track.appearance_intervals)
    assert uow.video_appearance_intervals.intervals[0].track_id == track.track_id

    assert len(uow.video_track_samples.samples) == 1
    link = uow.video_track_samples.samples[0]
    assert link.track_id == track.track_id
    assert link.sample_id == sample_id
    assert link.quality_score > 0.0

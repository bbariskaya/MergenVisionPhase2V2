"""Video result service current-projection unit tests."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, Self

import pytest

from app.application.ports.repositories import (
    FaceIdentityRepository,
    VideoAppearanceIntervalRepository,
    VideoTrackRepository,
)
from app.application.ports.unit_of_work import UnitOfWork
from app.application.services.video_result_service import VideoResultService
from app.domain.entities.face_identity import FaceIdentity
from app.domain.entities.video_job import VideoJob
from app.domain.entities.video_track import VideoAppearanceInterval, VideoTrack
from app.domain.value_objects import FaceId, JobId, ProcessId, ResultId, VideoId
from app.infrastructure.uuid7 import generate_uuid7


@dataclass
class _FakeFaceIdentityRepo(FaceIdentityRepository):
    identities: dict[FaceId, FaceIdentity] = field(default_factory=dict)

    async def add(self, identity: FaceIdentity) -> None:
        self.identities[identity.face_id] = identity

    async def get_by_id(self, face_id: FaceId) -> FaceIdentity | None:
        return self.identities.get(face_id)

    async def get_active_by_id(self, face_id: FaceId) -> FaceIdentity | None:
        identity = self.identities.get(face_id)
        return identity if identity and identity.is_active else None

    async def get_many_by_ids(self, face_ids: Any) -> list[FaceIdentity]:
        return [self.identities[fid] for fid in face_ids if fid in self.identities]

    async def update(self, identity: FaceIdentity) -> None:
        self.identities[identity.face_id] = identity

    async def update_with_expected_version(
        self, identity: FaceIdentity, expected_version: int
    ) -> FaceIdentity:
        raise NotImplementedError

    async def list_all(self) -> Any:
        return list(self.identities.values())

    async def search(self, query: str | None = None, status: str | None = None, is_active: bool = True) -> Any:
        raise NotImplementedError


@dataclass
class _FakeVideoTrackRepo(VideoTrackRepository):
    tracks: list[VideoTrack] = field(default_factory=list)

    async def add(self, track: VideoTrack) -> None:
        self.tracks.append(track)

    async def get_by_id(self, track_id: uuid.UUID) -> VideoTrack | None:
        for t in self.tracks:
            if t.track_id == track_id:
                return t
        return None

    async def list_by_job_id(self, job_id: JobId) -> list[VideoTrack]:
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
class _FakeAppearanceIntervalRepo(VideoAppearanceIntervalRepository):
    intervals: list[VideoAppearanceInterval] = field(default_factory=list)

    async def add(self, interval: VideoAppearanceInterval) -> None:
        self.intervals.append(interval)

    async def list_by_track_id(self, track_id: uuid.UUID) -> list[VideoAppearanceInterval]:
        return [iv for iv in self.intervals if iv.track_id == track_id]

    async def delete_by_job_id(self, job_id: JobId) -> int:
        before = len(self.intervals)
        self.intervals = [iv for iv in self.intervals if iv.job_id != job_id]
        return before - len(self.intervals)


@dataclass
class _FakeVideoJobRepo:
    jobs: dict[JobId, VideoJob] = field(default_factory=dict)

    async def get_by_id(self, job_id: JobId) -> VideoJob | None:
        return self.jobs.get(job_id)


@dataclass
class _FakeUoW(UnitOfWork):
    video_jobs: _FakeVideoJobRepo = field(default_factory=_FakeVideoJobRepo)
    video_tracks: _FakeVideoTrackRepo = field(default_factory=_FakeVideoTrackRepo)
    video_appearance_intervals: _FakeAppearanceIntervalRepo = field(default_factory=_FakeAppearanceIntervalRepo)
    face_identities: _FakeFaceIdentityRepo = field(default_factory=_FakeFaceIdentityRepo)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass

    async def commit(self) -> None:
        pass

    async def rollback(self) -> None:
        pass

    async def flush(self) -> None:
        pass


def _make_job(job_id: JobId) -> VideoJob:
    return VideoJob(
        job_id=job_id,
        video_id=VideoId(generate_uuid7()),
        process_id=ProcessId(generate_uuid7()),
        state="completed",
    )


def _make_track(job_id: JobId, face_id: FaceId) -> VideoTrack:
    return VideoTrack(
        track_id=generate_uuid7(),
        job_id=job_id,
        track_ordinal=0,
        face_id=face_id,
        recognition_result_id=ResultId(generate_uuid7()),
        status_at_processing="anonymous",
        name_at_processing=None,
        metadata_at_processing={},
        identity_version_at_processing=1,
        match_confidence=0.0,
        top1_score=None,
        top2_score=None,
        margin_score=None,
        threshold_used=None,
        first_frame_index=0,
        last_frame_index=10,
        first_pts_ns=0,
        last_pts_ns=100,
        total_duration_ns=100,
        detection_count=5,
        tracklet_count=1,
        best_sample_id=None,
    )


@pytest.mark.anyio
async def test_current_projection_reflects_updated_identity() -> None:
    job_id = JobId(generate_uuid7())
    face_id = FaceId(generate_uuid7())

    job = _make_job(job_id)
    track = _make_track(job_id, face_id)

    identity = FaceIdentity(
        face_id=face_id,
        status="known",
        is_active=True,
        display_name="Alice",
        identity_metadata={"department": "IT"},
        version=2,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    jobs_repo = _FakeVideoJobRepo()
    jobs_repo.jobs[job_id] = job

    uow = _FakeUoW(
        video_jobs=jobs_repo,
        video_tracks=_FakeVideoTrackRepo(tracks=[track]),
        face_identities=_FakeFaceIdentityRepo(identities={face_id: identity}),
    )

    service = VideoResultService(lambda: uow)
    people = await service.list_people(str(job_id))

    assert len(people) == 1
    summary = people[0]
    assert summary.status == "anonymous"
    assert summary.name is None
    assert summary.current_status == "known"
    assert summary.current_name == "Alice"
    assert summary.current_metadata == {"department": "IT"}

"""Read-side service for completed video job results."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from app.application.ports.unit_of_work import UnitOfWorkFactory
from app.domain.errors import JobNotFoundError
from app.domain.value_objects import JobId


@dataclass(frozen=True)
class PersonSummary:
    track_id: UUID
    face_id: UUID
    status: str
    name: str | None
    first_frame_index: int
    last_frame_index: int
    first_pts_ns: int
    last_pts_ns: int
    total_duration_ns: int
    detection_count: int
    appearance_count: int
    match_confidence: float


@dataclass(frozen=True)
class AppearanceSummary:
    track_id: UUID
    face_id: UUID
    start_frame_index: int
    end_frame_index: int
    start_pts_ns: int
    end_pts_ns: int
    detection_count: int


@dataclass(frozen=True)
class TimelineRecord:
    track_id: UUID
    face_id: UUID
    start_frame_index: int
    end_frame_index: int
    start_pts_ns: int
    end_pts_ns: int


class VideoResultService:
    def __init__(self, unit_of_work_factory: UnitOfWorkFactory) -> None:
        self._unit_of_work_factory = unit_of_work_factory

    async def list_people(self, job_id_str: str) -> list[PersonSummary]:
        job_id = self._parse_job_id(job_id_str)
        async with self._unit_of_work_factory() as uow:
            job = await uow.video_jobs.get_by_id(job_id)
            if job is None:
                raise JobNotFoundError(f"Job {job_id_str} not found")
            tracks = await uow.video_tracks.list_by_job_id(job_id)
            summaries: list[PersonSummary] = []
            for track in tracks:
                appearances = await uow.video_appearance_intervals.list_by_track_id(track.track_id)
                summaries.append(
                    PersonSummary(
                        track_id=track.track_id,
                        face_id=track.face_id,
                        status=track.status_at_processing,
                        name=track.name_at_processing,
                        first_frame_index=track.first_frame_index,
                        last_frame_index=track.last_frame_index,
                        first_pts_ns=track.first_pts_ns,
                        last_pts_ns=track.last_pts_ns,
                        total_duration_ns=track.total_duration_ns,
                        detection_count=track.detection_count,
                        appearance_count=len(appearances),
                        match_confidence=track.match_confidence,
                    )
                )
            return summaries

    async def list_appearances(self, job_id_str: str) -> list[AppearanceSummary]:
        job_id = self._parse_job_id(job_id_str)
        async with self._unit_of_work_factory() as uow:
            job = await uow.video_jobs.get_by_id(job_id)
            if job is None:
                raise JobNotFoundError(f"Job {job_id_str} not found")
            tracks = await uow.video_tracks.list_by_job_id(job_id)
            summaries: list[AppearanceSummary] = []
            for track in tracks:
                appearances = await uow.video_appearance_intervals.list_by_track_id(track.track_id)
                for interval in appearances:
                    summaries.append(
                        AppearanceSummary(
                            track_id=track.track_id,
                            face_id=track.face_id,
                            start_frame_index=interval.start_frame_index,
                            end_frame_index=interval.end_frame_index,
                            start_pts_ns=interval.start_pts_ns,
                            end_pts_ns=interval.end_pts_ns,
                            detection_count=interval.detection_count,
                        )
                    )
            return sorted(summaries, key=lambda a: (a.start_pts_ns, a.track_id.int))

    async def get_timeline(self, job_id_str: str) -> list[TimelineRecord]:
        appearances = await self.list_appearances(job_id_str)
        return [
            TimelineRecord(
                track_id=a.track_id,
                face_id=a.face_id,
                start_frame_index=a.start_frame_index,
                end_frame_index=a.end_frame_index,
                start_pts_ns=a.start_pts_ns,
                end_pts_ns=a.end_pts_ns,
            )
            for a in appearances
        ]

    @staticmethod
    def _parse_job_id(job_id_str: str) -> JobId:
        try:
            return JobId(UUID(job_id_str))
        except ValueError as exc:
            raise JobNotFoundError(f"Job id {job_id_str!r} is not a valid UUID") from exc

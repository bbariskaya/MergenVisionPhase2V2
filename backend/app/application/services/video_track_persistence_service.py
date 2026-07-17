"""Persist canonical video tracks, tracklets and appearances to PostgreSQL."""

from __future__ import annotations

import uuid
from collections.abc import Callable

from app.application.ports.unit_of_work import UnitOfWorkFactory
from app.application.services.video_identity_resolution_service import TrackIdentityOutcome
from app.domain.entities.video_track import (
    VideoAppearanceInterval,
    VideoTrack,
    VideoTracklet,
    VideoTrackSample,
)
from app.domain.entities.video_tracking import AppearanceInterval, CanonicalTrack
from app.domain.errors import IdentityResolutionError
from app.domain.value_objects import JobId

IdGenerator = Callable[[], uuid.UUID]


class VideoTrackPersistenceService:
    def __init__(
        self,
        unit_of_work_factory: UnitOfWorkFactory,
        id_generator: IdGenerator = uuid.uuid4,
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._id_generator = id_generator

    async def persist_tracks(
        self,
        job_id: JobId,
        track_outcomes: list[tuple[CanonicalTrack, TrackIdentityOutcome]],
    ) -> list[VideoTrack]:
        persisted: list[VideoTrack] = []
        async with self._unit_of_work_factory() as uow:
            # Make re-processing the same job idempotent: clear any partial
            # results left by a previous failed attempt before re-inserting.
            await uow.video_timeline_chunks.delete_by_job_id(job_id)
            await uow.video_appearance_intervals.delete_by_job_id(job_id)
            await uow.video_tracklets.delete_by_job_id(job_id)
            await uow.video_track_samples.delete_by_job_id(job_id)
            await uow.video_tracks.delete_by_job_id(job_id)
            await uow.flush()

            for track_ordinal, (track, outcome) in enumerate(track_outcomes):
                identity = await uow.face_identities.get_active_by_id(outcome.face_id)
                name_at_processing = None
                metadata_at_processing: dict[str, object] = {}
                identity_version = 1
                if identity is not None:
                    identity_version = identity.version
                    if identity.status == "known" and identity.display_name:
                        name_at_processing = identity.display_name
                    metadata_at_processing = dict(identity.identity_metadata or {})

                if outcome.result_id is None:
                    raise IdentityResolutionError(
                        f"Track {track.track_id} has no recognition_result_id"
                    )

                first_frame, last_frame, first_pts, last_pts = self._track_time_bounds(track)
                total_duration = self._total_duration(track.appearance_intervals)
                detection_count = sum(len(t.detections) for t in track.tracklets)
                tracklet_count = len(track.tracklets)
                max_quality = self._max_quality(track)

                video_track = VideoTrack(
                    track_id=track.track_id,
                    job_id=job_id,
                    track_ordinal=track_ordinal,
                    face_id=outcome.face_id,
                    recognition_result_id=outcome.result_id,
                    status_at_processing=outcome.status,
                    name_at_processing=name_at_processing,
                    metadata_at_processing=metadata_at_processing,
                    identity_version_at_processing=identity_version,
                    match_confidence=outcome.match_confidence,
                    top1_score=outcome.top1_score,
                    top2_score=outcome.top2_score,
                    margin_score=outcome.margin_score,
                    threshold_used=outcome.threshold_used,
                    first_frame_index=first_frame,
                    last_frame_index=last_frame,
                    first_pts_ns=first_pts,
                    last_pts_ns=last_pts,
                    total_duration_ns=total_duration,
                    detection_count=detection_count,
                    tracklet_count=tracklet_count,
                    best_sample_id=outcome.sample_id,
                )
                await uow.video_tracks.add(video_track)
                persisted.append(video_track)
                await uow.flush()

                for tracklet_ordinal, tracklet in enumerate(track.tracklets):
                    valid_embeddings = sum(1 for d in tracklet.detections if d.embedding)
                    video_tracklet = VideoTracklet(
                        tracklet_id=tracklet.tracklet_id,
                        job_id=job_id,
                        track_id=track.track_id,
                        tracklet_ordinal=tracklet_ordinal,
                        first_frame_index=tracklet.detections[0].frame_index,
                        last_frame_index=tracklet.detections[-1].frame_index,
                        first_pts_ns=tracklet.detections[0].pts_ns,
                        last_pts_ns=tracklet.detections[-1].pts_ns,
                        observation_count=len(tracklet.detections),
                        valid_embedding_count=valid_embeddings,
                        state=tracklet.state,
                        mean_quality=tracklet.mean_quality,
                        max_quality=tracklet.max_quality,
                    )
                    await uow.video_tracklets.add(video_tracklet)

                for interval_index, interval in enumerate(track.appearance_intervals):
                    video_interval = VideoAppearanceInterval(
                        appearance_id=self._id_generator(),
                        job_id=job_id,
                        track_id=track.track_id,
                        interval_index=interval_index,
                        start_frame_index=interval.start_frame_index,
                        end_frame_index=interval.end_frame_index,
                        start_pts_ns=interval.start_pts_ns,
                        end_pts_ns=interval.end_pts_ns,
                        detection_count=interval.detection_count,
                    )
                    await uow.video_appearance_intervals.add(video_interval)

                if outcome.sample_id is not None:
                    link = VideoTrackSample(
                        track_id=track.track_id,
                        sample_id=outcome.sample_id,
                        sample_rank=0,
                        quality_score=max_quality if max_quality is not None else 0.0,
                        purpose="identity_seed",
                    )
                    await uow.video_track_samples.add(link)

            await uow.commit()
        return persisted

    @staticmethod
    def _track_time_bounds(track: CanonicalTrack) -> tuple[int, int, int, int]:
        if track.appearance_intervals:
            first_interval = min(track.appearance_intervals, key=lambda i: i.start_frame_index)
            last_interval = max(track.appearance_intervals, key=lambda i: i.end_frame_index)
            return (
                first_interval.start_frame_index,
                last_interval.end_frame_index,
                first_interval.start_pts_ns,
                last_interval.end_pts_ns,
            )
        if track.tracklets:
            first_tracklet = min(track.tracklets, key=lambda t: t.detections[0].frame_index)
            last_tracklet = max(track.tracklets, key=lambda t: t.detections[-1].frame_index)
            return (
                first_tracklet.detections[0].frame_index,
                last_tracklet.detections[-1].frame_index,
                first_tracklet.detections[0].pts_ns,
                last_tracklet.detections[-1].pts_ns,
            )
        raise IdentityResolutionError(f"Track {track.track_id} has no time bounds")

    @staticmethod
    def _total_duration(intervals: list[AppearanceInterval]) -> int:
        return sum(i.end_pts_ns - i.start_pts_ns for i in intervals)

    @staticmethod
    def _max_quality(track: CanonicalTrack) -> float | None:
        values = [
            t.max_quality for t in track.tracklets if t.max_quality is not None
        ]
        return max(values) if values else None

"""Public overlay artifact generation and reading for video timeline."""

from __future__ import annotations

import gzip
import hashlib
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from app.application.ports.object_store import ObjectStore
from app.application.ports.unit_of_work import UnitOfWorkFactory
from app.application.services.video_identity_resolution_service import (
    TrackIdentityOutcome,
)
from app.domain.entities.video_timeline_chunk import VideoTimelineChunk
from app.domain.entities.video_tracking import CanonicalTrack
from app.domain.value_objects import FaceId, JobId

IdGenerator = Callable[[], uuid.UUID]

OVERLAY_SCHEMA_VERSION = "public_overlay_v1"
COMPRESSION = "gzip"
RECORDS_PER_CHUNK = 1000


@dataclass(frozen=True)
class OverlayDetection:
    track_id: uuid.UUID
    face_id: FaceId
    status: str
    name: str | None
    bbox: dict[str, int]
    confidence: float


@dataclass(frozen=True)
class OverlayFrame:
    frame_index: int
    pts_ns: int
    detections: list[OverlayDetection]


class VideoOverlayService:
    def __init__(
        self,
        *,
        unit_of_work_factory: UnitOfWorkFactory,
        object_store: ObjectStore,
        bucket_name: str,
        id_generator: IdGenerator = uuid.uuid4,
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._object_store = object_store
        self._bucket_name = bucket_name
        self._id_generator = id_generator

    async def write_public_overlay(
        self,
        job_id: JobId,
        video_id: uuid.UUID,
        canonical_tracks: list[CanonicalTrack],
        outcomes: list[TrackIdentityOutcome],
    ) -> VideoTimelineChunk:
        if len(canonical_tracks) != len(outcomes):
            raise ValueError("canonical_tracks and outcomes must have same length")

        track_to_outcome = {track.track_id: outcome for track, outcome in zip(canonical_tracks, outcomes, strict=True)}
        name_by_face_id = await self._resolve_current_names(
            {outcome.face_id for outcome in outcomes}
        )

        frames_by_index: dict[int, tuple[int, list[dict[str, Any]]]] = {}
        for track in canonical_tracks:
            outcome = track_to_outcome[track.track_id]
            for tracklet in track.tracklets:
                for detection in tracklet.detections:
                    record: dict[str, Any] = {
                        "track_id": str(track.track_id),
                        "face_id": str(outcome.face_id),
                        "status": outcome.status,
                        "name": name_by_face_id.get(outcome.face_id),
                        "bbox": {
                            "x": detection.bbox.x,
                            "y": detection.bbox.y,
                            "width": detection.bbox.width,
                            "height": detection.bbox.height,
                        },
                        "confidence": round(detection.detector_score, 4),
                        "provenance": "detected",
                    }
                    existing = frames_by_index.get(detection.frame_index)
                    if existing is None:
                        frames_by_index[detection.frame_index] = (
                            detection.pts_ns,
                            [record],
                        )
                    else:
                        existing[1].append(record)

        sorted_records = [
            {"frame_index": frame_index, "pts_ns": pts_ns, "detections": detections}
            for frame_index, (pts_ns, detections) in sorted(frames_by_index.items())
        ]

        object_key = f"videos/{video_id}/jobs/{job_id}/timeline/{0:06d}.jsonl.gz"
        payload = "\n".join(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) for rec in sorted_records).encode("utf-8")
        compressed = gzip.compress(payload, compresslevel=6)
        sha256 = hashlib.sha256(compressed).hexdigest()

        stat = await self._object_store.upload(
            object_key,
            compressed,
            "application/x-ndjson+gzip",
        )

        first_pts_ns = cast(int, sorted_records[0]["pts_ns"]) if sorted_records else 0
        last_pts_ns = cast(int, sorted_records[-1]["pts_ns"]) if sorted_records else 0

        chunk = VideoTimelineChunk(
            chunk_id=self._id_generator(),
            job_id=job_id,
            artifact_kind="public_overlay",
            sequence_no=0,
            start_pts_ns=first_pts_ns,
            end_pts_ns=last_pts_ns,
            bucket=stat.bucket,
            object_key=stat.key,
            content_sha256=sha256,
            size_bytes=len(compressed),
            record_count=len(sorted_records),
            schema_version=OVERLAY_SCHEMA_VERSION,
            compression=COMPRESSION,
            created_at=datetime.now(UTC),
        )

        async with self._unit_of_work_factory() as uow:
            await uow.video_timeline_chunks.add(chunk)
            await uow.commit()

        return chunk

    async def read_overlay_frames(
        self,
        job_id: JobId,
        start_pts_ns: int | None = None,
        end_pts_ns: int | None = None,
    ) -> list[dict[str, Any]]:
        async with self._unit_of_work_factory() as uow:
            chunks = await uow.video_timeline_chunks.list_by_job_id(
                job_id,
                artifact_kind="public_overlay",
            )

        results: list[dict[str, Any]] = []
        for chunk in chunks:
            if end_pts_ns is not None and chunk.start_pts_ns > end_pts_ns:
                continue
            if start_pts_ns is not None and chunk.end_pts_ns < start_pts_ns:
                continue
            data = await self._object_store.get(chunk.object_key)
            if data is None:
                continue
            decompressed = gzip.decompress(data)
            for line in decompressed.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                pts = int(record["pts_ns"])
                if start_pts_ns is not None and pts < start_pts_ns:
                    continue
                if end_pts_ns is not None and pts > end_pts_ns:
                    continue
                results.append(record)
        return results

    async def _resolve_current_names(self, face_ids: set[FaceId]) -> dict[FaceId, str | None]:
        names: dict[FaceId, str | None] = {}
        async with self._unit_of_work_factory() as uow:
            for face_id in face_ids:
                identity = await uow.face_identities.get_by_id(face_id)
                names[face_id] = identity.display_name if identity else None
        return names

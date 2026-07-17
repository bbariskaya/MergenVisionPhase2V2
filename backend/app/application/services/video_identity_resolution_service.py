"""Canonical video-track identity resolution reusing the image lifecycle."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass

from app.application.ports.vector_store import VectorCandidate
from app.application.services.identity_storage_lifecycle_service import (
    IdentityStorageLifecycleService,
    RecognitionOutcome,
)
from app.domain.entities.video_tracking import AppearanceInterval, CanonicalTrack, TrackDetection
from app.domain.errors import IdentityResolutionError
from app.domain.value_objects import BoundingBox, FaceId, ProcessId, ResultId, SampleId


@dataclass(frozen=True)
class TrackIdentityOutcome:
    track_id: uuid.UUID
    face_id: FaceId
    sample_id: SampleId | None
    result_id: ResultId | None
    status: str
    match_confidence: float
    top1_score: float | None
    top2_score: float | None
    margin_score: float | None
    threshold_used: float | None


class VideoIdentityResolutionService:
    def __init__(
        self,
        lifecycle: IdentityStorageLifecycleService,
        match_threshold: float = 0.95,
        candidate_limit: int = 5,
        margin_multiplier: float = 0.95,
        max_template_samples: int = 3,
        min_template_gap: int = 2,
    ) -> None:
        self._lifecycle = lifecycle
        self._match_threshold = match_threshold
        self._candidate_limit = candidate_limit
        self._margin_multiplier = margin_multiplier
        self._max_template_samples = max_template_samples
        self._min_template_gap = min_template_gap

    async def resolve(
        self,
        process_id: ProcessId,
        canonical_tracks: list[CanonicalTrack],
        crop_bytes_by_track_id: dict[uuid.UUID, bytes],
    ) -> list[TrackIdentityOutcome]:
        assigned: dict[uuid.UUID, list[AppearanceInterval]] = {}
        outcomes: list[TrackIdentityOutcome] = []

        for track in canonical_tracks:
            embedding = self._canonical_embedding(track)
            bbox = self._representative_bbox(track)
            candidates = sorted(
                await self._lifecycle.query_candidates(embedding, self._candidate_limit),
                key=lambda c: c.score,
                reverse=True,
            )
            top1_score = self._clamp_score(candidates[0].score) if candidates else None
            top2_score = self._clamp_score(candidates[1].score) if len(candidates) > 1 else None
            margin_score = (
                (top1_score - top2_score)
                if top1_score is not None and top2_score is not None
                else None
            )

            outcome = await self._resolve_one(
                process_id=process_id,
                track=track,
                embedding=embedding,
                bbox=bbox,
                candidates=candidates,
                top1_score=top1_score,
                top2_score=top2_score,
                assigned=assigned,
                crop_bytes_by_track_id=crop_bytes_by_track_id,
            )
            outcomes.append(
                TrackIdentityOutcome(
                    track_id=track.track_id,
                    face_id=outcome.face_id,
                    sample_id=outcome.sample_id,
                    result_id=outcome.result_id,
                    status=outcome.status,
                    match_confidence=self._clamp_score(outcome.match_confidence),
                    top1_score=top1_score,
                    top2_score=top2_score,
                    margin_score=margin_score,
                    threshold_used=self._match_threshold,
                )
            )
            assigned.setdefault(outcome.face_id, []).extend(track.appearance_intervals)

        return outcomes

    async def _resolve_one(
        self,
        process_id: ProcessId,
        track: CanonicalTrack,
        embedding: tuple[float, ...],
        bbox: BoundingBox,
        candidates: list[VectorCandidate],
        top1_score: float | None,
        top2_score: float | None,
        assigned: dict[uuid.UUID, list[AppearanceInterval]],
        crop_bytes_by_track_id: dict[uuid.UUID, bytes],
    ) -> RecognitionOutcome:
        accepted = await self._try_accept_candidate(
            process_id=process_id,
            bbox=bbox,
            candidates=candidates,
            top1_score=top1_score,
            top2_score=top2_score,
            track_intervals=track.appearance_intervals,
            assigned=assigned,
        )
        if accepted is not None:
            return accepted

        crop_bytes = crop_bytes_by_track_id.get(track.track_id)
        if crop_bytes is None:
            raise IdentityResolutionError(f"No crop bytes provided for track {track.track_id}")

        confidence = 0.0
        if top1_score is not None:
            confidence = top1_score
        return await self._lifecycle.create_new_identity_for_process(
            process_id=process_id,
            crop_bytes=crop_bytes,
            embedding=embedding,
            bbox=bbox,
            match_confidence=confidence,
        )

    async def _try_accept_candidate(
        self,
        process_id: ProcessId,
        bbox: BoundingBox,
        candidates: list[VectorCandidate],
        top1_score: float | None,
        top2_score: float | None,
        track_intervals: list[AppearanceInterval],
        assigned: dict[uuid.UUID, list[AppearanceInterval]],
    ) -> RecognitionOutcome | None:
        if top1_score is None or top1_score < self._match_threshold:
            return None
        if top2_score is not None and top2_score >= top1_score * self._margin_multiplier:
            return None

        for candidate in candidates:
            if candidate.score < self._match_threshold:
                break
            if self._blocked(candidate.face_id, track_intervals, assigned):
                continue
            try:
                return await self._lifecycle.accept_candidate_for_process(
                    process_id=process_id,
                    candidate=candidate,
                    bbox=bbox,
                )
            except IdentityResolutionError:
                continue
        return None

    def _blocked(
        self,
        face_id: FaceId,
        track_intervals: list[AppearanceInterval],
        assigned: dict[uuid.UUID, list[AppearanceInterval]],
    ) -> bool:
        for existing in assigned.get(face_id, []):
            for interval in track_intervals:
                if self._intervals_overlap(existing, interval):
                    return True
        return False

    @staticmethod
    def _intervals_overlap(a: AppearanceInterval, b: AppearanceInterval) -> bool:
        return not (a.end_frame_index < b.start_frame_index or b.end_frame_index < a.start_frame_index)

    def _canonical_embedding(self, track: CanonicalTrack) -> tuple[float, ...]:
        selected: list[TrackDetection] = []
        for tracklet in track.tracklets:
            for det in sorted(tracklet.detections, key=lambda d: -d.quality_score):
                if not det.embedding or len(det.embedding) != 512:
                    continue
                if all(abs(det.frame_index - s.frame_index) >= self._min_template_gap for s in selected):
                    selected.append(det)
                if len(selected) >= self._max_template_samples:
                    break
            if len(selected) >= self._max_template_samples:
                break

        if not selected:
            for tracklet in track.tracklets:
                for det in tracklet.detections:
                    if det.embedding and len(det.embedding) == 512:
                        selected.append(det)
                        break
                if selected:
                    break

        if not selected:
            raise IdentityResolutionError(f"No valid embedding for track {track.track_id}")

        weights = [max(0.0, d.quality_score) for d in selected]
        weight_sum = sum(weights)
        if weight_sum == 0.0:
            weights = [1.0] * len(selected)
            weight_sum = float(len(selected))

        dim = len(selected[0].embedding)
        centroid = [0.0] * dim
        for det, weight in zip(selected, weights, strict=False):
            for i, v in enumerate(det.embedding):
                centroid[i] += v * weight
        centroid = [v / weight_sum for v in centroid]
        return self._l2_normalize(tuple(centroid))

    @staticmethod
    def _l2_normalize(embedding: tuple[float, ...]) -> tuple[float, ...]:
        norm = math.sqrt(sum(v * v for v in embedding))
        if norm == 0.0:
            raise IdentityResolutionError("Embedding norm is zero")
        if not math.isfinite(norm):
            raise IdentityResolutionError("Embedding norm is not finite")
        return tuple(v / norm for v in embedding)

    @staticmethod
    def _clamp_score(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _representative_bbox(track: CanonicalTrack) -> BoundingBox:
        if not track.tracklets:
            return BoundingBox(x=0, y=0, width=1, height=1)
        for tracklet in track.tracklets:
            if tracklet.detections:
                best = max(tracklet.detections, key=lambda d: d.quality_score)
                return best.bbox
        first = track.tracklets[0].detections[0]
        return first.bbox

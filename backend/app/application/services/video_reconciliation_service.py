"""Conservative raw-track reconciliation into canonical tracks."""

from __future__ import annotations

import math
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from app.domain.entities.video_tracking import (
    AppearanceInterval,
    CanonicalTrack,
    RawTracklet,
)

IdGenerator = Callable[[], uuid.UUID]


@dataclass(frozen=True)
class _Interval:
    start: int
    end: int


class VideoReconciliationService:
    def __init__(
        self,
        merge_threshold: float = 0.6,
        id_generator: IdGenerator = uuid.uuid4,
    ) -> None:
        self.merge_threshold = merge_threshold
        self._id_generator = id_generator

    def reconcile_tracklets(
        self,
        tracklets: list[RawTracklet],
    ) -> list[CanonicalTrack]:
        sorted_tracklets = sorted(
            tracklets,
            key=lambda t: t.detections[0].frame_index if t.detections else 0,
        )
        tracks: list[CanonicalTrack] = []

        for tracklet in sorted_tracklets:
            candidate_scores: list[tuple[int, float]] = []
            rep = self._representative_embedding(tracklet)
            for idx, track in enumerate(tracks):
                if self._overlaps(track, tracklet):
                    continue
                if track.track_id in tracklet.__dict__.get("cannot_link_track_ids", set()):
                    continue
                score = self._cosine_similarity(rep, track.representative_embedding)
                if score >= self.merge_threshold:
                    candidate_scores.append((idx, score))

            if candidate_scores:
                candidate_scores.sort(key=lambda p: -p[1])
                best_idx = candidate_scores[0][0]
                tracks[best_idx].tracklets.append(tracklet)
                tracks[best_idx].representative_embedding = self._mean_embedding(
                    [tracks[best_idx].representative_embedding, rep]
                )
            else:
                track = CanonicalTrack(
                    track_id=self._id_generator(),
                    tracklets=[tracklet],
                    representative_embedding=rep,
                )
                tracks.append(track)

        for track in tracks:
            track.appearance_intervals = self._build_appearance_intervals(track.tracklets)

        return tracks

    def _representative_embedding(self, tracklet: RawTracklet) -> tuple[float, ...]:
        embs = [d.embedding for d in tracklet.detections if d.embedding]
        if not embs:
            return ()
        return self._mean_embedding(embs)

    def _mean_embedding(self, embeddings: list[tuple[float, ...]]) -> tuple[float, ...]:
        if not embeddings:
            return ()
        dim = len(embeddings[0])
        sums = [0.0] * dim
        for emb in embeddings:
            for i, v in enumerate(emb):
                sums[i] += v
        denom = float(len(embeddings))
        return tuple(s / denom for s in sums)

    def _cosine_similarity(
        self,
        a: tuple[float, ...],
        b: tuple[float, ...],
    ) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _overlaps(self, track: CanonicalTrack, tracklet: RawTracklet) -> bool:
        if not tracklet.detections:
            return False
        t_start = tracklet.detections[0].frame_index
        t_end = tracklet.detections[-1].frame_index
        for other in track.tracklets:
            if not other.detections:
                continue
            o_start = other.detections[0].frame_index
            o_end = other.detections[-1].frame_index
            if t_start <= o_end and o_start <= t_end:
                return True
        return False

    def _build_appearance_intervals(
        self,
        tracklets: list[RawTracklet],
    ) -> list[AppearanceInterval]:
        if not tracklets:
            return []
        merged: list[AppearanceInterval] = []
        for tracklet in sorted(tracklets, key=lambda t: t.detections[0].frame_index):
            detections = tracklet.detections
            interval = AppearanceInterval(
                start_frame_index=detections[0].frame_index,
                end_frame_index=detections[-1].frame_index,
                start_pts_ns=detections[0].pts_ns,
                end_pts_ns=detections[-1].pts_ns,
                detection_count=len(detections),
            )
            if merged and merged[-1].end_frame_index >= interval.start_frame_index - 1:
                prev = merged[-1]
                merged[-1] = AppearanceInterval(
                    start_frame_index=prev.start_frame_index,
                    end_frame_index=max(prev.end_frame_index, interval.end_frame_index),
                    start_pts_ns=prev.start_pts_ns,
                    end_pts_ns=max(prev.end_pts_ns, interval.end_pts_ns),
                    detection_count=prev.detection_count + interval.detection_count,
                )
            else:
                merged.append(interval)
        return merged

"""Metadata-only tracker: builds raw temporal tracklets from observations."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass

from app.application.ports.video_observations import FaceObservation, VideoObservationFrame
from app.domain.entities.video_tracking import RawTracklet, TrackDetection, TrackletTemplate
from app.domain.value_objects import BoundingBox

IdGenerator = Callable[[], uuid.UUID]


@dataclass(frozen=True)
class _ActiveTracklet:
    tracklet: RawTracklet
    last_frame_index: int
    last_bbox: tuple[int, int, int, int]


class VideoTrackingService:
    def __init__(
        self,
        max_gap_frames: int = 5,
        iou_threshold: float = 0.3,
        id_generator: IdGenerator = uuid.uuid4,
    ) -> None:
        self.max_gap_frames = max_gap_frames
        self.iou_threshold = iou_threshold
        self._id_generator = id_generator

    def build_raw_tracklets(
        self,
        frames: list[VideoObservationFrame],
        *,
        job_id: uuid.UUID | None = None,
    ) -> list[RawTracklet]:
        closed: list[RawTracklet] = []
        active: list[_ActiveTracklet] = []
        next_ordinal = 0
        effective_job_id = job_id or self._id_generator()

        for frame in sorted(frames, key=lambda f: (f.frame_index, f.pts_ns)):
            detections = [d for d in frame.detections if d.tracking_eligible]

            survivors: list[_ActiveTracklet] = []
            for at in active:
                gap = frame.frame_index - at.last_frame_index - 1
                if gap > self.max_gap_frames:
                    at.tracklet.state = "lost"
                    closed.append(at.tracklet)
                else:
                    survivors.append(at)
            active = survivors

            used_active: set[int] = set()
            used_detection: set[int] = set()
            matches: list[tuple[int, float, int]] = []

            for ai, at in enumerate(active):
                for di, det in enumerate(detections):
                    iou = _iou(at.last_bbox, _bbox_tuple(det.bbox))
                    if iou >= self.iou_threshold:
                        matches.append((ai, iou, di))

            matches.sort(key=lambda m: (-m[1], m[0], m[2]))
            for ai, _, di in matches:
                if ai in used_active or di in used_detection:
                    continue
                used_active.add(ai)
                used_detection.add(di)
                det = detections[di]
                at = active[ai]
                at.tracklet.detections.append(_to_track_detection(det, frame))
                active[ai] = _ActiveTracklet(
                    tracklet=at.tracklet,
                    last_frame_index=frame.frame_index,
                    last_bbox=_bbox_tuple(det.bbox),
                )

            for di, det in enumerate(detections):
                if di in used_detection:
                    continue
                tracklet = RawTracklet(
                    tracklet_id=self._id_generator(),
                    job_id=effective_job_id,
                    ordinal=next_ordinal,
                    state="confirmed",
                    detections=[_to_track_detection(det, frame)],
                )
                next_ordinal += 1
                active.append(
                    _ActiveTracklet(
                        tracklet=tracklet,
                        last_frame_index=frame.frame_index,
                        last_bbox=_bbox_tuple(det.bbox),
                    )
                )

        for at in active:
            at.tracklet.state = "lost"
            closed.append(at.tracklet)

        for tracklet in closed:
            qualities = [d.quality_score for d in tracklet.detections]
            tracklet.mean_quality = sum(qualities) / len(qualities) if qualities else None
            tracklet.max_quality = max(qualities) if qualities else None

        return sorted(closed, key=lambda t: t.ordinal)

    def select_tracklet_template(
        self,
        tracklet: RawTracklet,
        *,
        max_samples: int = 5,
        min_frame_gap: int = 2,
    ) -> TrackletTemplate:
        max_samples = max(1, max_samples)
        indexed = sorted(
            enumerate(tracklet.detections),
            key=lambda p: (-p[1].quality_score, p[1].frame_index),
        )
        selected: list[tuple[int, TrackDetection]] = []
        for idx, det in indexed:
            if all(abs(det.frame_index - s[1].frame_index) >= min_frame_gap for s in selected):
                selected.append((idx, det))
            if len(selected) >= max_samples:
                break
        selected.sort(key=lambda p: p[1].frame_index)
        indices = [p[0] for p in selected]
        qualities = [p[1].quality_score for p in selected]
        return TrackletTemplate(
            tracklet_id=tracklet.tracklet_id,
            sample_indices=indices,
            qualities=qualities,
        )


def _to_track_detection(det: FaceObservation, frame: VideoObservationFrame) -> TrackDetection:
    return TrackDetection(
        detection_id=det.detection_id,
        frame_index=frame.frame_index,
        pts_ns=frame.pts_ns,
        bbox=det.bbox,
        landmarks=det.landmarks,
        detector_score=det.detector_score,
        quality_score=det.quality_score,
        embedding=det.embedding,
        raw_track_key=det.raw_track_key,
    )


def _bbox_tuple(bbox: BoundingBox) -> tuple[int, int, int, int]:
    return (bbox.x, bbox.y, bbox.width, bbox.height)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    xi1, yi1 = max(ax, bx), max(ay, by)
    xi2, yi2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

"""ByteTrack-style IoU tracker for face observations.

Adapted from FoundationVision/ByteTrack
(https://github.com/FoundationVision/ByteTrack), yolox/tracker/byte_tracker.py,
commit d1bf0191adff59bc8fcfeaa0b33d3d1642552a99, MIT license.

Local changes:
- Operates on ``FaceObservation`` objects rather than raw tensors.
- No Cython/lap dependencies; uses project geometry and SciPy assignment.
- Appearance-free; embedding data is stored only for downstream template aggregation.
- Scene-cut reset and dual lost-timeout criteria (frames + nanoseconds).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mergenvision_video_lab.contracts import FaceObservation, RawTrackletSummary, TrackAssignment
from mergenvision_video_lab.tracking.assignment import (
    _GATED_COST,
    fuse_score,
    gated_iou_distance,
    linear_assignment,
)
from mergenvision_video_lab.tracking.base import TrackState
from mergenvision_video_lab.tracking.geometry import (
    bbox_xyxy_to_tlwh,
    tlwh_to_tlbr,
    tlwh_to_xyah,
)
from mergenvision_video_lab.tracking.kalman import KalmanFilter


@dataclass
class Detection:
    """Temporary detection wrapper consumed by the tracker."""

    observation_id: str
    frame_index: int
    pts_ns: int
    tlwh: np.ndarray
    score: float
    embedding: np.ndarray | None = None
    quality: float = 0.0

    @property
    def tlbr(self) -> np.ndarray:
        return tlwh_to_tlbr(self.tlwh)


class Tracklet:
    """One raw tracklet with a Kalman filter and observation history."""

    _id_counter = 0

    def __init__(
        self,
        strategy: str,
        _id_allocator: Callable[[], int] | None = None,
    ) -> None:
        if _id_allocator is not None:
            tracklet_id = _id_allocator()
        else:
            Tracklet._id_counter += 1
            tracklet_id = Tracklet._id_counter
        self.raw_tracklet_id = f"RT{tracklet_id:06d}"
        self.strategy = strategy
        self.state = TrackState.New
        self.is_activated = False
        self.mean: np.ndarray | None = None
        self.covariance: np.ndarray | None = None
        self.kalman_filter: KalmanFilter | None = None

        self.observation_ids: list[str] = []
        self.frame_indices: list[int] = []
        self.pts_ns_list: list[int] = []
        self.bbox_xyxy_history: list[tuple[float, float, float, float]] = []
        self.scores: list[float] = []
        self.embeddings: list[np.ndarray] = []
        self.qualities: list[float] = []

        # Appearance-template evidence (used by the hybrid tracker).
        self._appearance_embs: list[np.ndarray] = []
        self._appearance_qualities: list[float] = []
        self._appearance_pts_ns: list[int] = []

        self.start_frame_index: int | None = None
        self.start_pts_ns: int | None = None
        self.last_frame_index: int | None = None
        self.last_pts_ns: int | None = None
        self.time_since_update = 0
        self.tracklet_len = 0

    @classmethod
    def reset_id_counter(cls) -> None:
        cls._id_counter = 0

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            # Should only happen before activation; fall back to last history.
            if self.bbox_xyxy_history:
                return bbox_xyxy_to_tlwh(np.asarray(self.bbox_xyxy_history[-1]))
            return np.zeros(4, dtype=np.float64)
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2.0
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        return tlwh_to_tlbr(self.tlwh)

    def predict(self) -> None:
        if self.kalman_filter is None or self.mean is None or self.covariance is None:
            return
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def activate(self, kalman_filter: KalmanFilter, frame_index: int, pts_ns: int) -> None:
        self.kalman_filter = kalman_filter
        self.mean, self.covariance = kalman_filter.initiate(tlwh_to_xyah(self.tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.start_frame_index = frame_index
        self.start_pts_ns = pts_ns
        self.last_frame_index = frame_index
        self.last_pts_ns = pts_ns
        self.tracklet_len = 0
        self.time_since_update = 0

    def re_activate(self, det: Detection, frame_index: int, pts_ns: int) -> None:
        if self.kalman_filter is None or self.mean is None or self.covariance is None:
            return
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, tlwh_to_xyah(det.tlwh)
        )
        self._record(det, frame_index, pts_ns)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.time_since_update = 0
        self.tracklet_len = 0

    def update(self, det: Detection, frame_index: int, pts_ns: int) -> None:
        if self.kalman_filter is None or self.mean is None or self.covariance is None:
            return
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, tlwh_to_xyah(det.tlwh)
        )
        self._record(det, frame_index, pts_ns)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.time_since_update = 0
        self.tracklet_len += 1

    def _record(self, det: Detection, frame_index: int, pts_ns: int) -> None:
        self.observation_ids.append(det.observation_id)
        self.frame_indices.append(frame_index)
        self.pts_ns_list.append(pts_ns)
        self.bbox_xyxy_history.append(
            (float(det.tlbr[0]), float(det.tlbr[1]), float(det.tlbr[2]), float(det.tlbr[3]))
        )
        self.scores.append(det.score)
        self.qualities.append(det.quality)
        if det.embedding is not None:
            self.embeddings.append(det.embedding.copy())
        self.last_frame_index = frame_index
        self.last_pts_ns = pts_ns
        self.score = det.score

    def update_appearance(
        self,
        embedding: np.ndarray,
        quality: float,
        pts_ns: int,
        top_k: int,
        min_separation_ns: int,
    ) -> None:
        """Update the short-term appearance template with a new embedding."""
        if embedding is None or embedding.size == 0:
            return
        embedding = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm <= 0.0 or not np.isfinite(norm):
            return
        embedding = embedding / norm

        self._appearance_embs.append(embedding)
        self._appearance_qualities.append(float(quality))
        self._appearance_pts_ns.append(pts_ns)

        candidates = sorted(
            zip(
                self._appearance_qualities,
                self._appearance_pts_ns,
                self._appearance_embs,
                strict=False,
            ),
            key=lambda x: x[0],
            reverse=True,
        )
        selected: list[tuple[float, int, np.ndarray]] = []
        for q, pts, emb in candidates:
            if all(abs(pts - s_pts) >= min_separation_ns for _, s_pts, _ in selected):
                selected.append((q, pts, emb))
            if len(selected) >= top_k:
                break

        self._appearance_qualities = [q for q, _, _ in selected]
        self._appearance_pts_ns = [pts for _, pts, _ in selected]
        self._appearance_embs = [emb for _, _, emb in selected]

    @property
    def appearance_template(self) -> np.ndarray | None:
        """Return the quality-weighted, L2-normalized appearance centroid."""
        if not self._appearance_embs:
            return None
        weights = np.asarray(self._appearance_qualities, dtype=np.float64)
        embs = np.asarray(self._appearance_embs, dtype=np.float32)
        if np.sum(weights) <= 0.0:
            return None
        centroid = np.average(embs, axis=0, weights=weights)
        norm = float(np.linalg.norm(centroid))
        if norm <= 0.0 or not np.isfinite(norm):
            return None
        return (centroid / norm).astype(np.float32)

    def mark_lost(self) -> None:
        self.state = TrackState.Lost
        self.time_since_update += 1

    def mark_removed(self) -> None:
        self.state = TrackState.Removed

    def tick_lost(self) -> None:
        # Lost-timeout is computed from frame/PTS gaps; this counter is kept
        # for symmetry with the upstream tracklet but not double-incremented.
        pass

    def to_summary(self) -> RawTrackletSummary:
        return RawTrackletSummary(
            raw_tracklet_id=self.raw_tracklet_id,
            strategy=self.strategy,
            first_frame_index=self.start_frame_index or -1,
            last_frame_index=self.last_frame_index or -1,
            first_pts_ns=self.start_pts_ns or -1,
            last_pts_ns=self.last_pts_ns or -1,
            observation_count=len(self.observation_ids),
            detection_ordinal_ids=self.observation_ids.copy(),
            state="active" if self.state == TrackState.Tracked else "removed",
        )


def _observation_to_detection(
    obs: FaceObservation,
    embeddings: np.ndarray,
) -> Detection:
    tlwh = bbox_xyxy_to_tlwh(np.asarray(obs.bbox_xyxy.to_list()))
    emb: np.ndarray | None = None
    if obs.embedding_index is not None and 0 <= obs.embedding_index < embeddings.shape[0]:
        emb = embeddings[obs.embedding_index]
    return Detection(
        observation_id=obs.observation_id,
        frame_index=obs.frame_index,
        pts_ns=obs.pts_ns,
        tlwh=tlwh,
        score=float(obs.detector_score),
        embedding=emb,
        quality=float(obs.quality.composite_quality_score),
    )


class ByteTrackIoUTracker:
    """ByteTrack lifecycle with IoU-only association for face metadata."""

    strategy = "byte_iou"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.frame_index = -1
        self.tracked_tracklets: list[Tracklet] = []
        self.lost_tracklets: list[Tracklet] = []
        self._removed_tracklets: list[Tracklet] = []
        self.kalman_filter = KalmanFilter()
        self._seen_observation_ids: set[str] = set()
        self._seen_frame_indices: set[int] = set()
        self._next_tracklet_id = 0

    def _allocate_tracklet_id(self) -> int:
        self._next_tracklet_id += 1
        return self._next_tracklet_id

    def _max_lost_frames(self) -> int:
        return int(self.config["max_lost_frames"])

    def _max_lost_ns(self) -> int:
        return int(self.config["max_lost_ns"])

    def _scene_cut_reset(self) -> bool:
        return bool(self.config.get("scene_cut_reset", True))

    def _order_detections(self, detections: list[Detection]) -> list[Detection]:
        """Return detections in the deterministic observation order."""
        return sorted(detections, key=lambda d: (d.frame_index, d.observation_id))

    def update(
        self,
        frame_index: int,
        pts_ns: int,
        observations: Sequence[FaceObservation],
        embeddings: np.ndarray,
        scene_cut_before: bool,
    ) -> Sequence[TrackAssignment]:
        if frame_index in self._seen_frame_indices:
            raise ValueError(f"duplicate frame_index in tracker update: {frame_index}")
        if frame_index < self.frame_index:
            raise ValueError(f"frame_index regression: {frame_index} < {self.frame_index}")
        self.frame_index = frame_index
        self._seen_frame_indices.add(frame_index)

        if scene_cut_before and self._scene_cut_reset():
            self._reset_all_tracklets()

        detections = [
            _observation_to_detection(obs, embeddings)
            for obs in observations
            if obs.tracking_eligible
        ]
        detections = self._order_detections(detections)

        for det in detections:
            if det.observation_id in self._seen_observation_ids:
                raise ValueError(f"duplicate observation_id: {det.observation_id}")
            self._seen_observation_ids.add(det.observation_id)

        high_thresh = float(self.config["high_detection_threshold"])
        low_thresh = float(self.config["low_detection_threshold"])
        new_thresh = float(self.config["new_track_threshold"])

        dets_high = [d for d in detections if d.score >= high_thresh]
        dets_low = [d for d in detections if low_thresh <= d.score < high_thresh]

        # Separate activated and unconfirmed tracked tracklets.
        tracked_activated: list[Tracklet] = []
        unconfirmed: list[Tracklet] = []
        for t in self.tracked_tracklets:
            if t.is_activated:
                tracked_activated.append(t)
            else:
                unconfirmed.append(t)

        # Predict all tracks in the association pool.
        strack_pool = tracked_activated + self.lost_tracklets
        for t in strack_pool:
            t.predict()

        # ---------- First association: activated + lost vs high-score detections ----------
        first_matches, u_track, u_det = self._first_stage_association(strack_pool, dets_high)
        activated: list[Tracklet] = []
        refind: list[Tracklet] = []
        for ti, di in first_matches:
            track = strack_pool[ti]
            det = dets_high[di]
            if track.state == TrackState.Tracked:
                track.update(det, frame_index, pts_ns)
            else:
                track.re_activate(det, frame_index, pts_ns)
                refind.append(track)
            self._after_track_update(track, det)
            activated.append(track)

        # ---------- Second association: unmatched tracked vs low-score detections ----------
        r_tracked = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        second_matches, u_track2, _ = self._associate(
            r_tracked,
            dets_low,
            min_iou=float(self.config["second_stage_min_iou"]),
            use_fuse_score=False,
        )
        for ti, di in second_matches:
            track = r_tracked[ti]
            det = dets_low[di]
            if track.state == TrackState.Tracked:
                track.update(det, frame_index, pts_ns)
            else:
                track.re_activate(det, frame_index, pts_ns)
                refind.append(track)
            self._after_track_update(track, det)
            activated.append(track)

        for t in r_tracked:
            if t not in {r_tracked[m[0]] for m in second_matches}:
                t.mark_lost()
                self.lost_tracklets.append(t)

        # ---------- Unconfirmed tracks vs remaining high-score detections ----------
        remaining_high = [dets_high[i] for i in u_det]
        unconf_matches, u_unconfirmed, u_det2 = self._associate(
            unconfirmed,
            remaining_high,
            min_iou=float(self.config["unconfirmed_min_iou"]),
            use_fuse_score=True,
        )
        for ti, di in unconf_matches:
            track = unconfirmed[ti]
            det = remaining_high[di]
            track.update(det, frame_index, pts_ns)
            self._after_track_update(track, det)
            activated.append(track)
        for i in u_unconfirmed:
            unconfirmed[i].mark_removed()
            self._removed_tracklets.append(unconfirmed[i])

        # ---------- Initialize new tracklets from unmatched high-score detections ----------
        final_unmatched_high = [remaining_high[i] for i in u_det2]
        for det in final_unmatched_high:
            if det.score >= new_thresh:
                track = Tracklet(
                    strategy=self.strategy,
                    _id_allocator=self._allocate_tracklet_id,
                )
                track._record(det, frame_index, pts_ns)
                track.activate(self.kalman_filter, frame_index, pts_ns)
                self._after_track_update(track, det)
                activated.append(track)

        # ---------- Update track lists ----------
        self.tracked_tracklets = [
            t for t in self.tracked_tracklets if t.state == TrackState.Tracked
        ]
        self.tracked_tracklets = self._joint_stracks(self.tracked_tracklets, activated)
        self.tracked_tracklets = self._joint_stracks(self.tracked_tracklets, refind)

        self.lost_tracklets = [t for t in self.lost_tracklets if t.state != TrackState.Removed]
        self.lost_tracklets = self._sub_stracks(self.lost_tracklets, self.tracked_tracklets)
        for t in self.lost_tracklets:
            t.tick_lost()
        # Mark timed-out lost tracks as removed.
        still_lost: list[Tracklet] = []
        for t in self.lost_tracklets:
            last_frame = t.last_frame_index if t.last_frame_index is not None else frame_index
            last_pts = t.last_pts_ns if t.last_pts_ns is not None else pts_ns
            frame_gap = frame_index - last_frame
            ns_gap = pts_ns - last_pts
            if frame_gap > self._max_lost_frames() or ns_gap > self._max_lost_ns():
                t.mark_removed()
                self._removed_tracklets.append(t)
            else:
                still_lost.append(t)
        self.lost_tracklets = still_lost

        # Build assignments for this frame.
        assignments: list[TrackAssignment] = []
        assigned_ids: set[str] = set()
        for track in self.tracked_tracklets:
            if track.observation_ids and track.last_frame_index == frame_index:
                obs_id = track.observation_ids[-1]
                if obs_id not in assigned_ids:
                    assignments.append(
                        TrackAssignment(
                            observation_id=obs_id,
                            frame_index=frame_index,
                            pts_ns=pts_ns,
                            raw_tracklet_id=track.raw_tracklet_id,
                            strategy=self.strategy,
                        )
                    )
                    assigned_ids.add(obs_id)

        # Consistency: no duplicate raw IDs in this frame and no observation assigned twice.
        frame_track_ids = [a.raw_tracklet_id for a in assignments]
        if len(frame_track_ids) != len(set(frame_track_ids)):
            raise ValueError("duplicate raw_tracklet_id within one frame")
        if len({a.observation_id for a in assignments}) != len(assignments):
            raise ValueError("one observation assigned to multiple tracklets")

        return assignments

    def _after_track_update(self, track: Tracklet, det: Detection) -> None:
        """Hook called after a track is updated or activated."""
        return

    def _first_stage_association(
        self,
        strack_pool: list[Tracklet],
        dets_high: list[Detection],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """IoU + fuse_score association for activated/lost tracks and high detections."""
        return self._associate(
            strack_pool,
            dets_high,
            min_iou=float(self.config["first_stage_min_iou"]),
            use_fuse_score=True,
        )

    def _associate(
        self,
        tracks: list[Tracklet],
        detections: list[Detection],
        min_iou: float,
        use_fuse_score: bool,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        track_boxes = [t.tlbr for t in tracks]
        det_boxes = [d.tlbr for d in detections]
        cost = gated_iou_distance(track_boxes, det_boxes, min_iou)
        if use_fuse_score:
            scores = np.array([d.score for d in detections])
            cost = fuse_score(cost, scores)
            # Ensure gated entries remain gated after fuse_score.
            cost[cost > _GATED_COST / 2] = _GATED_COST

        matches, u_track, u_det = linear_assignment(cost, thresh=1.0)
        return matches, u_track, u_det

    def _joint_stracks(self, a: list[Tracklet], b: list[Tracklet]) -> list[Tracklet]:
        exists = {t.raw_tracklet_id: t for t in a}
        res = a.copy()
        for t in b:
            if t.raw_tracklet_id not in exists:
                res.append(t)
                exists[t.raw_tracklet_id] = t
        return res

    def _sub_stracks(self, a: list[Tracklet], b: list[Tracklet]) -> list[Tracklet]:
        b_ids = {t.raw_tracklet_id for t in b}
        return [t for t in a if t.raw_tracklet_id not in b_ids]

    def _reset_all_tracklets(self) -> None:
        for t in self.tracked_tracklets + self.lost_tracklets:
            t.mark_removed()
            self._removed_tracklets.append(t)
        self.tracked_tracklets = []
        self.lost_tracklets = []

    def finalize(self) -> None:
        for t in self.tracked_tracklets + self.lost_tracklets:
            t.mark_removed()
            self._removed_tracklets.append(t)
        self.tracked_tracklets = []
        self.lost_tracklets = []

    def active_tracklet_ids(self) -> list[str]:
        return [t.raw_tracklet_id for t in self.tracked_tracklets]

    def removed_tracklets(self) -> list[Tracklet]:
        return self._removed_tracklets

# Phase 2 M6–M8 Video Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the Phase 2 video recognition product: Python tracking/reconciliation, canonical identity resolution, worker/job E2E, result/timeline/playback API, and client overlay; make the listed Makefile targets green and produce the SPRINT-003 review package.

**Architecture:** A metadata-only Python tracker consumes compact observations (either protobuf/zstd from the native DeepStream worker or synthetic test observations), builds raw temporal tracklets, reconciles them into canonical tracks via embedding similarity, resolves each canonical track to a persistent face identity through the existing Qdrant/PG lifecycle, persists track/appearance/timeline artifacts, and exposes paginated overlay metadata over the existing FastAPI contract.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy (async), Qdrant, MinIO, pytest, no OpenCV/PIL production decode.

## Global Constraints

- Backend first / API-first; UI consumes versioned API only.
- Python metadata tracker is the first implementation; C++ rewrite only with profiling evidence.
- Tracker does NOT own persistent identity; identity resolution goes through `IdentityStorageLifecycleService`.
- Frame/PTS order must be preserved; batch boundaries do not reset track state.
- Bounding boxes are original display-space integer pixels.
- `faceId`, `sampleId`, `trackId`, `trackletId`, `jobId`, `processId` are UUIDv7.
- Product output is original video + time-synchronized overlay metadata (not annotated MP4).
- No full-frame CPU decode on the production hot path; best-shot crop extraction is an explicit post-processing gate.

---

## Subsystem A — M6 Tracking & Reconciliation

### Files created/modified

- Create `backend/app/application/ports/video_observations.py` — lightweight observation DTOs matching `video_observation_v1.proto`.
- Create `backend/app/domain/entities/video_tracking.py` — `TrackDetection`, `RawTracklet`, `TrackletTemplate`, `AppearanceInterval`.
- Create `backend/app/application/services/video_tracking_service.py` — build raw tracklets + select template samples.
- Create `backend/app/application/services/video_reconciliation_service.py` — conservative merge of raw tracklets into canonical tracks.
- Create `backend/tests/unit/services/test_video_tracking_service.py`.
- Create `backend/tests/unit/services/test_video_reconciliation_service.py`.
- Modify `Makefile` — add `phase2-m6-track-template` and `phase2-m6-track-reconcile` targets.

---

### Task 1: Define observation and tracking domain models

**Files:**
- Create: `backend/app/application/ports/video_observations.py`
- Create: `backend/app/domain/entities/video_tracking.py`

**Interfaces:**
- Consumes: `app.domain.value_objects.BoundingBox`
- Produces: `FaceObservation`, `VideoObservationFrame`, `TrackDetection`, `RawTracklet`, `TrackletTemplate`

- [ ] **Step 1: Write the failing import tests**

```python
# backend/tests/unit/services/test_video_tracking_service.py
from app.application.ports.video_observations import FaceObservation, VideoObservationFrame
from app.domain.entities.video_tracking import RawTracklet, TrackDetection
from app.application.services.video_tracking_service import VideoTrackingService


def test_service_can_be_constructed() -> None:
    svc = VideoTrackingService(max_gap_frames=5, iou_threshold=0.3)
    assert svc.max_gap_frames == 5
```

Run: `cd backend && python -m pytest tests/unit/services/test_video_tracking_service.py::test_service_can_be_constructed -v`
Expected: FAIL with import error.

- [ ] **Step 2: Create the observation DTOs**

```python
# backend/app/application/ports/video_observations.py
from __future__ import annotations

from dataclasses import dataclass

from app.domain.value_objects import BoundingBox


@dataclass(frozen=True)
class FaceObservation:
    detection_id: str
    ordinal: int
    bbox: BoundingBox
    landmarks: tuple[float, ...]
    detector_score: float
    quality_score: float
    tracking_eligible: bool
    recognition_eligible: bool
    rejection_code: str
    embedding: tuple[float, ...]
    model_version: str = ""
    preprocess_version: str = ""


@dataclass(frozen=True)
class VideoObservationFrame:
    job_id: str
    video_id: str
    stream_index: int
    frame_index: int
    source_pts: int
    pts_ns: int
    display_width: int
    display_height: int
    detections: tuple[FaceObservation, ...]
```

- [ ] **Step 3: Create the tracking domain entities**

```python
# backend/app/domain/entities/video_tracking.py
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from app.domain.value_objects import BoundingBox


@dataclass(frozen=True)
class TrackDetection:
    detection_id: str
    frame_index: int
    pts_ns: int
    bbox: BoundingBox
    landmarks: tuple[float, ...]
    detector_score: float
    quality_score: float
    embedding: tuple[float, ...]


@dataclass
class RawTracklet:
    tracklet_id: uuid.UUID
    job_id: uuid.UUID
    ordinal: int
    state: str = "confirmed"
    detections: list[TrackDetection] = field(default_factory=list)
    mean_quality: float | None = None
    max_quality: float | None = None

    def __post_init__(self) -> None:
        if self.state not in {"confirmed", "lost", "removed"}:
            raise ValueError(f"Invalid tracklet state: {self.state}")


@dataclass(frozen=True)
class TrackletTemplate:
    tracklet_id: uuid.UUID
    sample_indices: list[int]
    qualities: list[float]


@dataclass(frozen=True)
class AppearanceInterval:
    start_frame_index: int
    end_frame_index: int
    start_pts_ns: int
    end_pts_ns: int
    detection_count: int = 0
```

- [ ] **Step 4: Verify imports pass**

Run: `cd backend && python -m pytest tests/unit/services/test_video_tracking_service.py::test_service_can_be_constructed -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/application/ports/video_observations.py \
        backend/app/domain/entities/video_tracking.py \
        backend/tests/unit/services/test_video_tracking_service.py
git commit -m "feat(m6): observation DTOs and raw tracking domain entities"
```

---

### Task 2: Implement raw tracklet builder

**Files:**
- Create/modify: `backend/app/application/services/video_tracking_service.py`

**Interfaces:**
- Consumes: `VideoObservationFrame`
- Produces: `RawTracklet` ordered by first frame; assigns `tracklet_id` and `ordinal`.

**Algorithm:**
1. Keep a list of active tracklets (last detection frame and last bbox).
2. For each incoming frame, compute pairwise IoU between active tracklet bboxes and current detections.
3. Use greedy best-IoU bipartite matching. A match requires `iou >= iou_threshold`.
4. Unmatched detections start new tracklets.
5. Tracklets with no match for more than `max_gap_frames` are closed (`state = "lost"`).
6. Detections with `tracking_eligible == False` are skipped.

- [ ] **Step 1: Write a failing tracklet-count test**

```python
# backend/tests/unit/services/test_video_tracking_service.py
import uuid

from app.domain.value_objects import BoundingBox
from app.application.ports.video_observations import FaceObservation, VideoObservationFrame


def _det(x: int, frame: int = 0) -> FaceObservation:
    return FaceObservation(
        detection_id=f"d-{x}",
        ordinal=x,
        bbox=BoundingBox(x=x, y=0, width=10, height=10),
        landmarks=(0.0,) * 10,
        detector_score=0.9,
        quality_score=0.8,
        tracking_eligible=True,
        recognition_eligible=True,
        rejection_code="",
        embedding=(0.0,) * 512,
    )


def test_single_detection_per_frame_yields_one_tracklet() -> None:
    svc = VideoTrackingService(max_gap_frames=3, iou_threshold=0.3)
    frames = [
        VideoObservationFrame(
            job_id=str(uuid.uuid7()),
            video_id=str(uuid.uuid7()),
            stream_index=0,
            frame_index=i,
            source_pts=i,
            pts_ns=i * 33_000_000,
            display_width=640,
            display_height=480,
            detections=(_det(10 * i, i),),
        )
        for i in range(5)
    ]
    tracklets = svc.build_raw_tracklets(frames)
    assert len(tracklets) == 1
    assert len(tracklets[0].detections) == 5
```

Run: `cd backend && python -m pytest tests/unit/services/test_video_tracking_service.py::test_single_detection_per_frame_yields_one_tracklet -v`
Expected: FAIL (`build_raw_tracklets` not defined).

- [ ] **Step 2: Implement `VideoTrackingService.build_raw_tracklets`**

```python
# backend/app/application/services/video_tracking_service.py
from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.application.ports.video_observations import FaceObservation, VideoObservationFrame
from app.domain.entities.video_tracking import RawTracklet, TrackDetection


@dataclass(frozen=True)
class _ActiveTracklet:
    tracklet: RawTracklet
    last_frame_index: int
    last_bbox: tuple[int, int, int, int]


class VideoTrackingService:
    def __init__(self, max_gap_frames: int = 5, iou_threshold: float = 0.3) -> None:
        self.max_gap_frames = max_gap_frames
        self.iou_threshold = iou_threshold

    def build_raw_tracklets(
        self,
        frames: list[VideoObservationFrame],
        *,
        job_id: uuid.UUID | None = None,
    ) -> list[RawTracklet]:
        closed: list[RawTracklet] = []
        active: list[_ActiveTracklet] = []
        next_ordinal = 0

        for frame in sorted(frames, key=lambda f: (f.frame_index, f.pts_ns)):
            detections = [d for d in frame.detections if d.tracking_eligible]
            used_active = set()
            used_detection = set()
            matches: list[tuple[int, float, int]] = []

            for ai, at in enumerate(active):
                for di, det in enumerate(detections):
                    if di in used_detection:
                        continue
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
                at.tracklet.detections.append(self._to_track_detection(det, frame))
                active[ai] = _ActiveTracklet(
                    tracklet=at.tracklet,
                    last_frame_index=frame.frame_index,
                    last_bbox=_bbox_tuple(det.bbox),
                )

            for di, det in enumerate(detections):
                if di in used_detection:
                    continue
                tracklet_id = uuid.uuid7()
                tracklet = RawTracklet(
                    tracklet_id=tracklet_id,
                    job_id=job_id or uuid.uuid7(),
                    ordinal=next_ordinal,
                    state="confirmed",
                    detections=[self._to_track_detection(det, frame)],
                )
                next_ordinal += 1
                active.append(
                    _ActiveTracklet(
                        tracklet=tracklet,
                        last_frame_index=frame.frame_index,
                        last_bbox=_bbox_tuple(det.bbox),
                    )
                )

            still_active: list[_ActiveTracklet] = []
            for at in active:
                if frame.frame_index - at.last_frame_index > self.max_gap_frames:
                    at.tracklet.state = "lost"
                    closed.append(at.tracklet)
                else:
                    still_active.append(at)
            active = still_active

        for at in active:
            at.tracklet.state = "lost"
            closed.append(at.tracklet)

        for tracklet in closed:
            qualities = [d.quality_score for d in tracklet.detections]
            tracklet.mean_quality = sum(qualities) / len(qualities) if qualities else None
            tracklet.max_quality = max(qualities) if qualities else None

        return sorted(closed, key=lambda t: t.ordinal)

    def _to_track_detection(
        self, det: FaceObservation, frame: VideoObservationFrame
    ) -> TrackDetection:
        return TrackDetection(
            detection_id=det.detection_id,
            frame_index=frame.frame_index,
            pts_ns=frame.pts_ns,
            bbox=det.bbox,
            landmarks=det.landmarks,
            detector_score=det.detector_score,
            quality_score=det.quality_score,
            embedding=det.embedding,
        )


def _bbox_tuple(bbox) -> tuple[int, int, int, int]:
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
```

- [ ] **Step 3: Verify tests pass**

Run: `cd backend && python -m pytest tests/unit/services/test_video_tracking_service.py -v`
Expected: PASS.

- [ ] **Step 4: Add gap/occlusion test**

```python
def test_gap_within_max_gap_keeps_one_tracklet() -> None:
    svc = VideoTrackingService(max_gap_frames=2, iou_threshold=0.3)
    frames = [
        VideoObservationFrame(
            job_id=str(uuid.uuid7()),
            video_id=str(uuid.uuid7()),
            stream_index=0,
            frame_index=i,
            source_pts=i,
            pts_ns=i * 33_000_000,
            display_width=640,
            display_height=480,
            detections=(_det(100, i),) if i in (0, 3) else (),
        )
        for i in range(4)
    ]
    tracklets = svc.build_raw_tracklets(frames)
    assert len(tracklets) == 1


def test_gap_beyond_max_gap_splits_tracklet() -> None:
    svc = VideoTrackingService(max_gap_frames=1, iou_threshold=0.3)
    frames = [
        VideoObservationFrame(
            job_id=str(uuid.uuid7()),
            video_id=str(uuid.uuid7()),
            stream_index=0,
            frame_index=i,
            source_pts=i,
            pts_ns=i * 33_000_000,
            display_width=640,
            display_height=480,
            detections=(_det(100, i),) if i in (0, 3) else (),
        )
        for i in range(4)
    ]
    tracklets = svc.build_raw_tracklets(frames)
    assert len(tracklets) == 2
```

- [ ] **Step 5: Run tests and commit**

```bash
cd backend && python -m pytest tests/unit/services/test_video_tracking_service.py -v
git add backend/app/application/services/video_tracking_service.py \
        backend/tests/unit/services/test_video_tracking_service.py
git commit -m "feat(m6): raw tracklet builder with IoU association"
```

---

### Task 3: Implement tracklet quality/template selector

**Files:**
- Modify: `backend/app/application/services/video_tracking_service.py`
- Add method: `select_tracklet_template`

**Rules:**
- Choose up to `max_samples` detections from a tracklet.
- Prefer high `quality_score`, then temporal diversity (avoid consecutive frames).
- Return deterministic indices sorted by frame.

- [ ] **Step 1: Write failing template test**

```python
def test_template_selects_best_quality_detections() -> None:
    svc = VideoTrackingService(max_gap_frames=3, iou_threshold=0.3)
    det_low = _det(0, 0)._replace(quality_score=0.3)
    det_high = _det(0, 1)._replace(quality_score=0.95)
    frames = [
        VideoObservationFrame(
            job_id=str(uuid.uuid7()),
            video_id=str(uuid.uuid7()),
            stream_index=0,
            frame_index=0,
            source_pts=0,
            pts_ns=0,
            display_width=640,
            display_height=480,
            detections=(det_low,),
        ),
        VideoObservationFrame(
            job_id=str(uuid.uuid7()),
            video_id=str(uuid.uuid7()),
            stream_index=0,
            frame_index=1,
            source_pts=1,
            pts_ns=33_000_000,
            display_width=640,
            display_height=480,
            detections=(det_high,),
        ),
    ]
    tracklets = svc.build_raw_tracklets(frames)
    template = svc.select_tracklet_template(tracklets[0], max_samples=1)
    assert template.sample_indices == [1]
```

Expected: FAIL (`select_tracklet_template` missing).

- [ ] **Step 2: Implement selector**

```python
# Add to VideoTrackingService
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
```

- [ ] **Step 3: Verify tests**

Run: `cd backend && python -m pytest tests/unit/services/test_video_tracking_service.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add backend/app/application/services/video_tracking_service.py \
        backend/tests/unit/services/test_video_tracking_service.py
git commit -m "feat(m6): tracklet quality/template selector"
```

---

### Task 4: Implement conservative raw-track reconciliation

**Files:**
- Create: `backend/app/application/services/video_reconciliation_service.py`
- Create: `backend/tests/unit/services/test_video_reconciliation_service.py`

**Interfaces:**
- Consumes: `RawTracklet`
- Produces: `CanonicalTrack` with `track_id`, merged `tracklets`, non-overlapping `appearance_intervals`, and representative embedding.

**Algorithm:**
1. Sort tracklets by `first_pts_ns`.
2. For each tracklet, compute cosine similarity between its representative embedding (mean of selected template embeddings) and every existing canonical track.
3. Merge into the best-matching canonical track only if:
   - `similarity >= merge_threshold`;
   - the tracklet does not overlap in time with any tracklet already in that canonical track (cannot-link from co-occurrence);
   - the chosen track is not in the tracklet's cannot-link set.
4. Otherwise create a new canonical track.

- [ ] **Step 1: Define `CanonicalTrack` and write failing test**

```python
# backend/app/domain/entities/video_tracking.py (append)
@dataclass
class CanonicalTrack:
    track_id: uuid.UUID
    tracklets: list[RawTracklet] = field(default_factory=list)
    cannot_link_track_ids: set[uuid.UUID] = field(default_factory=set)
    representative_embedding: tuple[float, ...] = field(default_factory=tuple)
    appearance_intervals: list[AppearanceInterval] = field(default_factory=list)
```

```python
# backend/tests/unit/services/test_video_reconciliation_service.py
import uuid

from app.application.services.video_tracking_service import VideoTrackingService
from app.application.services.video_reconciliation_service import VideoReconciliationService
from app.domain.value_objects import BoundingBox
from app.application.ports.video_observations import FaceObservation, VideoObservationFrame


def _frame_with_embedding(
    frame_index: int,
    value: float,
    x: int = 100,
) -> VideoObservationFrame:
    emb = [0.0] * 512
    emb[0] = value
    return VideoObservationFrame(
        job_id=str(uuid.uuid7()),
        video_id=str(uuid.uuid7()),
        stream_index=0,
        frame_index=frame_index,
        source_pts=frame_index,
        pts_ns=frame_index * 33_000_000,
        display_width=640,
        display_height=480,
        detections=(
            FaceObservation(
                detection_id=f"d-{frame_index}",
                ordinal=0,
                bbox=BoundingBox(x=x, y=0, width=20, height=20),
                landmarks=(0.0,) * 10,
                detector_score=0.9,
                quality_score=0.8,
                tracking_eligible=True,
                recognition_eligible=True,
                rejection_code="",
                embedding=tuple(emb),
            ),
        ),
    )


def test_two_similar_non_overlapping_tracklets_merge() -> None:
    tracker = VideoTrackingService(max_gap_frames=1, iou_threshold=0.3)
    frames_a = [_frame_with_embedding(i, 1.0, x=10) for i in range(3)]
    frames_b = [_frame_with_embedding(i, 1.0, x=10) for i in range(5, 8)]
    tracklets = tracker.build_raw_tracklets(frames_a + frames_b)
    assert len(tracklets) == 2

    reconciler = VideoReconciliationService(merge_threshold=0.9)
    tracks = reconciler.reconcile_tracklets(tracklets)
    assert len(tracks) == 1
    assert len(tracks[0].tracklets) == 2
```

- [ ] **Step 2: Implement `VideoReconciliationService.reconcile_tracklets`**

```python
# backend/app/application/services/video_reconciliation_service.py
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass

from app.domain.entities.video_tracking import (
    AppearanceInterval,
    CanonicalTrack,
    RawTracklet,
    TrackDetection,
)


@dataclass(frozen=True)
class _Interval:
    start: int
    end: int


class VideoReconciliationService:
    def __init__(self, merge_threshold: float = 0.6) -> None:
        self.merge_threshold = merge_threshold

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
                    track_id=uuid.uuid7(),
                    tracklets=[tracklet],
                    representative_embedding=rep,
                )
                tracks.append(track)

        for track in tracks:
            track.appearance_intervals = self._build_appearance_intervals(track.tracklets)

        return tracks

    def _representative_embedding(self, tracklet: RawTracklet) -> tuple[float, ...]:
        if not tracklet.detections:
            return ()
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
        self, a: tuple[float, ...], b: tuple[float, ...]
    ) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
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
        self, tracklets: list[RawTracklet]
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
```

- [ ] **Step 3: Verify reconciliation tests**

Run: `cd backend && python -m pytest tests/unit/services/test_video_reconciliation_service.py -v`
Expected: PASS.

- [ ] **Step 4: Add cannot-link / different-people test**

```python
def test_overlapping_tracklets_do_not_merge() -> None:
    tracker = VideoTrackingService(max_gap_frames=10, iou_threshold=0.1)
    frames_ab = [
        VideoObservationFrame(
            job_id=str(uuid.uuid7()),
            video_id=str(uuid.uuid7()),
            stream_index=0,
            frame_index=i,
            source_pts=i,
            pts_ns=i * 33_000_000,
            display_width=640,
            display_height=480,
            detections=(
                _face(1.0, i, x=10),
                _face(-1.0, i, x=200),
            ),
        )
        for i in range(3)
    ]
    tracklets = tracker.build_raw_tracklets(frames_ab)
    assert len(tracklets) == 2

    reconciler = VideoReconciliationService(merge_threshold=0.9)
    tracks = reconciler.reconcile_tracklets(tracklets)
    assert len(tracks) == 2

def _face(value: float, frame_index: int, x: int) -> FaceObservation:
    emb = [0.0] * 512
    emb[0] = value
    return FaceObservation(
        detection_id=f"d-{frame_index}-{x}",
        ordinal=0,
        bbox=BoundingBox(x=x, y=0, width=20, height=20),
        landmarks=(0.0,) * 10,
        detector_score=0.9,
        quality_score=0.8,
        tracking_eligible=True,
        recognition_eligible=True,
        rejection_code="",
        embedding=tuple(emb),
    )
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/application/services/video_reconciliation_service.py \
        backend/app/domain/entities/video_tracking.py \
        backend/tests/unit/services/test_video_reconciliation_service.py
git commit -m "feat(m6): conservative raw-track reconciliation"
```

---

### Task 5: Wire Makefile targets for M6

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add targets**

Append to `Makefile` after `phase2-m6-native-full-observation`:

```makefile
phase2-m6-track-template: phase2-services
	@echo "==> phase2-m6-track-template: raw tracklet builder + quality template selector"
	cd $(BACKEND_DIR) && $(WITH_TEST_ENV) $(PYTEST) \
	    tests/unit/services/test_video_tracking_service.py -v
	@echo "==> phase2-m6-track-template passed"

phase2-m6-track-reconcile: phase2-services
	@echo "==> phase2-m6-track-reconcile: conservative raw-track reconciliation"
	cd $(BACKEND_DIR) && $(WITH_TEST_ENV) $(PYTEST) \
	    tests/unit/services/test_video_reconciliation_service.py -v
	@echo "==> phase2-m6-track-reconcile passed"
```

- [ ] **Step 2: Run both targets**

```bash
make phase2-m6-track-template
make phase2-m6-track-reconcile
```

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "build: phase2 m6 track template and reconcile make targets"
```

---

## Subsystem B — M7 Identity Resolution, Worker E2E & API

*Detailed sub-plan to be written after Subsystem A is merged. Scope summary:*

1. **`VideoIdentityResolutionService`** — takes a `CanonicalTrack`, runs quality-weighted search against Qdrant, resolves via `IdentityStorageLifecycleService`, persists `video_track`, `video_tracklet`, `appearance_interval`, `video_track_sample`. Handles `new_anonymous` creation with best-shot persistence.
2. **`VideoProcessingWorker`** — claims job from `VideoJobQueue`, spawns/scaffolds native observation extraction (or synthetic/fake adapter for the Python-only gate), reads observation chunks, runs tracking → reconciliation → identity resolution, updates job stage, writes timeline chunks to MinIO, writes result manifest, finalizes job.
3. **Result/Timeline/Playback API** — extend `videos.py` and `VideoUploadService` to return `GET /api/v1/videos/jobs/{jobId}/result` person summary, `GET /api/v1/videos/jobs/{jobId}/timeline` paginated overlay chunks, and `GET /api/v1/videos/{videoId}/playback` signed MinIO URL.

## Subsystem C — M8 Client Video Overlay

*Detailed sub-plan to be written after Subsystem B API contracts are frozen. Scope summary:*

1. Reusable `<VideoOverlayPlayer>` React component in the existing frontend baseline.
2. Fetches original video stream and timeline chunks separately.
3. Renders SVG/Canvas bounding boxes synchronised with `requestVideoFrameCallback().metadata.mediaTime`.
4. Does not bake identity names into detection records; uses mutable identity map.

## Subsystem D — Acceptance & Review Package

*Detailed sub-plan after M8. Scope summary:*

1. Orchestrate `make phase2-step0-static` → `make phase2-migrations` → `make phase2-m6-track-template` → `make phase2-m6-track-reconcile` → containerised native smoke → integration E2E.
2. Update `docs/implementation/CURRENT_SPRINT.md` and `IMPLEMENTATION_DETAILS.md`.
3. Produce `docs/implementation/review_packages/SPRINT-003-CODE-REVIEW-PACKAGE.md`.

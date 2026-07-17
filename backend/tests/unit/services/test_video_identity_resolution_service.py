"""M7 canonical video identity resolution unit tests."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import pytest

from app.application.ports.vector_store import VectorCandidate
from app.application.services.identity_storage_lifecycle_service import (
    IdentityStorageLifecycleService,
    RecognitionOutcome,
)
from app.application.services.video_identity_resolution_service import (
    VideoIdentityResolutionService,
)
from app.application.services.video_reconciliation_service import VideoReconciliationService
from app.application.services.video_tracking_service import VideoTrackingService
from app.domain.entities.video_tracking import CanonicalTrack
from app.domain.errors import IdentityResolutionError
from app.domain.value_objects import BoundingBox, FaceId, ProcessId, ResultId, SampleId
from app.infrastructure.uuid7 import generate_uuid7
from tests.fixtures.embedding_fixtures import DIMENSION

MATCH_THRESHOLD = 0.95


def _unit_vector(first_one_at: int) -> tuple[float, ...]:
    vector = [0.0] * DIMENSION
    vector[first_one_at] = 1.0
    return tuple(vector)


VECTOR_A = _unit_vector(0)
VECTOR_B = _unit_vector(1)
VECTOR_C = _unit_vector(2)


@dataclass
class _FakeIdentity:
    face_id: FaceId
    status: str = "anonymous"
    enrolled: bool = False


@dataclass
class _FakeLifecycle(IdentityStorageLifecycleService):
    identities: dict[int, _FakeIdentity] = field(default_factory=dict)
    next_face_index: int = 0
    outcomes: list[RecognitionOutcome] = field(default_factory=list)
    fail_accept_for: set[FaceId] = field(default_factory=set)

    async def query_candidates(self, embedding, top_k=None):  # type: ignore[override]
        key = self._vector_key(embedding)
        candidates = []
        for other_key, identity in self.identities.items():
            score = 1.0 if other_key == key else 0.0
            if score > 0.0:
                sample_id = SampleId(identity.face_id)
                candidates.append(VectorCandidate(sample_id, identity.face_id, score))
        return sorted(candidates, key=lambda c: c.score, reverse=True)[: top_k or 5]

    async def accept_candidate_for_process(self, process_id, candidate, bbox):  # type: ignore[override]
        identity = self.identities.get(self._vector_key_by_face(candidate.face_id))
        if identity is None or identity.face_id in self.fail_accept_for:
            raise IdentityResolutionError("Candidate rejected")
        outcome = RecognitionOutcome(
            process_id=process_id,
            face_id=identity.face_id,
            sample_id=SampleId(generate_uuid7()),
            result_id=ResultId(generate_uuid7()),
            status="known" if identity.enrolled else "anonymous",
            bounding_box=bbox,
            match_confidence=candidate.score,
        )
        self.outcomes.append(outcome)
        return outcome

    async def create_new_identity_for_process(  # type: ignore[override]
        self,
        process_id,
        crop_bytes,
        embedding,
        bbox,
        match_confidence=0.0,
    ):
        face_id = FaceId(generate_uuid7())
        key = self._vector_key(embedding)
        self.identities[key] = _FakeIdentity(face_id=face_id)
        outcome = RecognitionOutcome(
            process_id=process_id,
            face_id=face_id,
            sample_id=SampleId(generate_uuid7()),
            result_id=ResultId(generate_uuid7()),
            status="new_anonymous",
            bounding_box=bbox,
            match_confidence=match_confidence,
        )
        self.outcomes.append(outcome)
        return outcome

    def _vector_key(self, embedding: tuple[float, ...]) -> int:
        for i, v in enumerate(embedding):
            if abs(v) > 0.9:
                return i
        return -1

    def _vector_key_by_face(self, face_id: FaceId) -> int:
        for key, identity in self.identities.items():
            if identity.face_id == face_id:
                return key
        return -1

    async def resolve_or_create(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    async def resolve_or_create_for_process(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    async def add_sample(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    async def verify_identity_match(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    async def enroll_identity(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    async def deactivate_identity(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    async def delete_sample(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    async def start_process(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    async def complete_process(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    async def fail_process(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError


def _det(embedding: tuple[float, ...], frame_index: int, x: int = 10) -> any:
    from app.application.ports.video_observations import FaceObservation

    obs = FaceObservation(
        detection_id=f"d-{frame_index}",
        ordinal=0,
        bbox=BoundingBox(x=x, y=0, width=20, height=20),
        landmarks=(0.0,) * 10,
        detector_score=0.9,
        quality_score=0.9,
        tracking_eligible=True,
        recognition_eligible=True,
        rejection_code="",
        embedding=embedding,
        model_version="retinaface_r50_glintr100_v1",
        preprocess_version="cuda_nv12_align_v1",
    )
    return obs


def _build_canonical_track(embedding: tuple[float, ...], frames: list[int]) -> CanonicalTrack:
    tracker = VideoTrackingService(max_gap_frames=1, iou_threshold=0.3)
    from app.application.ports.video_observations import VideoObservationFrame

    observations = [
        VideoObservationFrame(
            job_id=str(generate_uuid7()),
            video_id=str(generate_uuid7()),
            stream_index=0,
            frame_index=f,
            source_pts=f,
            pts_ns=f * 33_000_000,
            display_width=640,
            display_height=480,
            detections=(_det(embedding, f),),
        )
        for f in frames
    ]
    tracklets = tracker.build_raw_tracklets(observations)
    reconciler = VideoReconciliationService(merge_threshold=0.9)
    tracks = reconciler.reconcile_tracklets(tracklets)
    return tracks[0]


def _crop_map(track: CanonicalTrack) -> dict[uuid.UUID, bytes]:
    return {track.track_id: b"fake-crop"}


async def test_first_unknown_track_returns_new_anonymous() -> None:
    lifecycle = _FakeLifecycle()
    service = VideoIdentityResolutionService(lifecycle)
    track = _build_canonical_track(VECTOR_A, [0, 1, 2])

    outcomes = await service.resolve(
        ProcessId(generate_uuid7()), [track], _crop_map(track)
    )

    assert len(outcomes) == 1
    assert outcomes[0].status == "new_anonymous"
    assert outcomes[0].top1_score is None


async def test_matching_existing_anonymous_returns_same_face_id() -> None:
    lifecycle = _FakeLifecycle()
    service = VideoIdentityResolutionService(lifecycle)
    track_a = _build_canonical_track(VECTOR_A, [0, 1, 2])
    first = await service.resolve(ProcessId(generate_uuid7()), [track_a], _crop_map(track_a))

    track_b = _build_canonical_track(VECTOR_A, [10, 11, 12])
    second = await service.resolve(ProcessId(generate_uuid7()), [track_b], _crop_map(track_b))

    assert first[0].status == "new_anonymous"
    assert second[0].status == "anonymous"
    assert second[0].face_id == first[0].face_id


async def test_known_match_after_enrollment() -> None:
    lifecycle = _FakeLifecycle()
    service = VideoIdentityResolutionService(lifecycle)
    track_a = _build_canonical_track(VECTOR_A, [0, 1, 2])
    first = await service.resolve(ProcessId(generate_uuid7()), [track_a], _crop_map(track_a))

    identity = lifecycle.identities[0]
    identity.enrolled = True
    identity.status = "known"

    track_b = _build_canonical_track(VECTOR_A, [10, 11, 12])
    second = await service.resolve(ProcessId(generate_uuid7()), [track_b], _crop_map(track_b))

    assert second[0].status == "known"
    assert second[0].face_id == first[0].face_id


async def test_overlapping_tracks_do_not_share_existing_face_id() -> None:
    lifecycle = _FakeLifecycle()
    service = VideoIdentityResolutionService(lifecycle)
    track_a = _build_canonical_track(VECTOR_A, [0, 1, 2])
    track_b = _build_canonical_track(VECTOR_A, [1, 2, 3])

    outcomes = await service.resolve(
        ProcessId(generate_uuid7()), [track_a, track_b], {track_a.track_id: b"c", track_b.track_id: b"c"}
    )

    assert len(outcomes) == 2
    assert outcomes[0].status == "new_anonymous"
    assert outcomes[1].face_id != outcomes[0].face_id
    assert outcomes[1].status == "new_anonymous"


async def test_close_top2_margin_creates_new_anonymous() -> None:
    lifecycle = _FakeLifecycle()
    identity_a = FaceId(generate_uuid7())
    identity_b = FaceId(generate_uuid7())
    lifecycle.identities[0] = _FakeIdentity(face_id=identity_a)
    lifecycle.identities[1] = _FakeIdentity(face_id=identity_b)

    async def _query(embedding, top_k=None):
        return [
            VectorCandidate(SampleId(identity_a), identity_a, 1.0),
            VectorCandidate(SampleId(identity_b), identity_b, 0.98),
        ]

    lifecycle.query_candidates = _query  # type: ignore[method-assign]
    service = VideoIdentityResolutionService(lifecycle, margin_multiplier=0.98)
    track = _build_canonical_track(VECTOR_A, [0, 1, 2])

    outcomes = await service.resolve(ProcessId(generate_uuid7()), [track], _crop_map(track))

    assert outcomes[0].status == "new_anonymous"
    assert outcomes[0].top2_score == pytest.approx(0.98)


async def test_missing_crop_bytes_raises() -> None:
    lifecycle = _FakeLifecycle()
    service = VideoIdentityResolutionService(lifecycle)
    track = _build_canonical_track(VECTOR_C, [0, 1, 2])

    with pytest.raises(IdentityResolutionError):
        await service.resolve(ProcessId(generate_uuid7()), [track], {})

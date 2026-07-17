# MergenVision Phase 1 + Phase 2 Implementation Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the full `ProjectRequirements.md` image API and the additive `videorequirements.md` video API on top of the existing Phase 1 Sprint 01 storage foundation.

**Architecture:** The existing foundation (`face_identity`, `face_sample`, `process_record`, `recognition_result` tables + PG/MinIO/Qdrant adapters + `IdentityStorageLifecycleService`) stays authoritative. Phase 1 adds a CPU ONNX inference adapter (RetinaFace + GlintR100), FastAPI endpoints, history/logging, and Docker packaging. Phase 2 layers video upload/retention, async job state, the native DeepStream/TensoRT GPU worker ported from `MergenVisionPhase2`, Python temporal tracking/reconciliation, person aggregation, and overlay metadata delivery. Identity matching and mutable identity state remain in the Python control plane for both phases.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2.0 async, Alembic, PostgreSQL 16, MinIO, Qdrant, ONNX Runtime (Phase 1 image inference), Docker. Phase 2 video path reuses DeepStream 9.0 / TensorRT native worker artifacts and tracking logic from `MergenVisionPhase2`.

## Global Constraints

- Image recognition must be `PASS` on real PG/MinIO/Qdrant storage before Phase 2 video recognition is claimed complete.
- `faceId` is immutable UUIDv7; only `status`/`display_name` are mutable through enroll/rename.
- Recognition statuses: `known`, `anonymous`, `new_anonymous` only.
- Bounding boxes returned in original image/video display resolution pixels.
- Video product output is original video + time-synchronized overlay metadata; annotated MP4 is debug/acceptance-only.
- No annotated-MP4 requirement may delay or block the overlay metadata pipeline.
- MinIO object finalization/validation must complete before a video job is queued.
- GPU worker is a compact-observation emitter; final identity reconciliation runs in Python.
- Idempotency-Key must prevent duplicate processes/identities/objects/vectors on retry.
- All config via environment variables; no hardcoded secrets or runtime paths in source.
- TDD discipline: failing test → minimal implementation → unit/integration → real service smoke → lint/type/build → review.

---

## Task 1: Phase 1 Sprint 02 — CPU ONNX Image Recognition Vertical Slice

**Files:**
- Create: `backend/app/application/ports/face_encoder.py`
- Create: `backend/app/infrastructure/inference/onnx_face_encoder.py`
- Create: `backend/app/application/services/image_recognition_service.py`
- Create: `backend/app/api/contracts.py`
- Create: `backend/app/api/main.py`
- Create: `backend/app/api/dependencies.py`
- Create: `backend/app/api/routes/faces.py`
- Create: `backend/app/api/routes/processes.py`
- Create: `backend/tests/unit/inference/test_onnx_face_encoder.py`
- Create: `backend/tests/integration/api/test_image_recognition_api.py`
- Modify: `backend/pyproject.toml`
- Modify: `backend/app/application/ports/__init__.py`
- Modify: `backend/app/infrastructure/persistence/sqlalchemy/repositories/process_record.py`
- Modify: `backend/app/infrastructure/persistence/sqlalchemy/repositories/recognition_result.py`
- Modify: `backend/app/infrastructure/persistence/sqlalchemy/repositories/face_identity.py`

**Interfaces:**
- Consumes: `IdentityStorageLifecycleService`, MinIO/Qdrant adapters, existing repository and entity contracts.
- Produces:
  - `FaceEncoderPort.detect_and_encode(image_bytes: bytes) -> list[FaceObservation]`
  - `ImageRecognitionService.recognize(image_bytes, match_threshold) -> ImageRecognitionResponse`
  - FastAPI routes matching `ProjectRequirements.md` sections 10–11.

- [ ] **Step 1: Add inference dependencies**
  Edit `backend/pyproject.toml` and append to `dependencies`:
  ```toml
  "onnxruntime>=1.18.0,<1.19.0",
  "opencv-python-headless>=4.10.0,<4.11.0",
  "pillow>=10.4.0,<10.5.0",
  "numpy>=1.26.0,<2.0.0",
  "fastapi>=0.111.0,<0.112.0",
  "uvicorn[standard]>=0.30.0,<0.31.0",
  "python-multipart>=0.0.9,<0.1.0",
  ```
  Run: `cd backend && uv sync` or `pip install -e .[dev]`.
  Expected: lock file refreshes, imports available.

- [ ] **Step 2: Define the inference port**
  Create `backend/app/application/ports/face_encoder.py`:
  ```python
  from __future__ import annotations

  from dataclasses import dataclass
  from typing import Protocol, Sequence

  from app.domain.value_objects import BoundingBox


  @dataclass(frozen=True)
  class FaceObservation:
      bounding_box: BoundingBox
      confidence: float
      embedding: Sequence[float]
      landmarks: Sequence[tuple[float, float]] | None = None


  class FaceEncoderPort(Protocol):
      async def detect_and_encode(self, image_bytes: bytes) -> list[FaceObservation]:
          ...
  ```

- [ ] **Step 3: Write a failing unit test for the encoder**
  Create `backend/tests/unit/inference/test_onnx_face_encoder.py`:
  ```python
  import pytest

  from app.infrastructure.inference.onnx_face_encoder import OnnxFaceEncoder


  @pytest.mark.asyncio
  async def test_detect_and_encode_returns_empty_for_blank_image(artifacts):
      encoder = OnnxFaceEncoder(
          detector_path=artifacts.retinaface_onnx,
          recognizer_path=artifacts.glintr100_onnx,
      )
      blank = b"\xff" * (64 * 64 * 3)  # invalid tiny image; decoder must reject
      with pytest.raises(ValueError):
          await encoder.detect_and_encode(blank)
  ```
  Run: `cd backend && pytest tests/unit/inference/test_onnx_face_encoder.py -v`
  Expected: FAIL — module/class not found.

- [ ] **Step 4: Implement the ONNX encoder adapter**
  Create `backend/app/infrastructure/inference/onnx_face_encoder.py`.
  It must:
  - Decode bytes with OpenCV/Pillow and reject corrupt/empty input.
  - Resize/pad to RetinaFace input 640x640, RGB, mean=[104,117,123].
  - Run `retinaface_r50_dynamic.onnx`; parse loc/conf/landms using priors and NMS (reference: `MergenVisionPhase2/backend/tests/native/detector_parity_lib.py:83-142`).
  - Map boxes/landmarks back to original image resolution.
  - For each detection, perform similarity-transform to the canonical 112x112 template from `MergenVisionPhase2/backend/native/configs/glintr100_preprocess_contract.json:36-42`.
  - Run `glintr100.onnx` (input name `input.1`, shape `[-1,3,112,112]`, RGB, `(pixel-127.5)/127.5`).
  - L2-normalize embeddings and return `FaceObservation` list.

- [ ] **Step 5: Verify the encoder unit test passes**
  Run: `cd backend && pytest tests/unit/inference/test_onnx_face_encoder.py -v`
  Expected: PASS.

- [ ] **Step 6: Write a real fixture integration test**
  Create `backend/tests/integration/api/test_image_recognition_api.py` with a real image fixture and assert:
  - `POST /faces/recognize` returns 200, `processId`, `faceCount`, and each face has `faceId`, `status`, `boundingBox`, `confidence`.
  - Empty face fixture returns 200 with `faceCount=0`.
  - Corrupt image returns structured 4xx error.

- [ ] **Step 7: Implement the image recognition service**
  Create `backend/app/application/services/image_recognition_service.py`:
  ```python
  class ImageRecognitionService:
      def __init__(self, encoder: FaceEncoderPort, lifecycle: IdentityStorageLifecycleService) -> None: ...
      async def recognize(self, image_bytes: bytes, match_threshold: float = 0.45) -> ImageRecognitionResponse: ...
  ```
  For each `FaceObservation`, call `lifecycle.resolve_or_create(crop_bytes, embedding, bbox, threshold)`.

- [ ] **Step 8: Implement FastAPI routes and contracts**
  Create Pydantic contracts in `backend/app/api/contracts.py` matching `ProjectRequirements.md` sections 10–11.
  Wire FastAPI in `backend/app/api/main.py`, `dependencies.py`, `routes/faces.py`, `routes/processes.py`.

- [ ] **Step 9: Implement history/detail repository queries**
  Add to existing repo files:
  - `FaceIdentityRepository.list_processes_for_face_id(face_id)` → process IDs and timestamps.
  - `RecognitionResultRepository.list_by_process_id(process_id)`.
  - `ProcessRecordRepository.get_by_id(process_id)` already exists; ensure it returns completed metadata.

- [ ] **Step 10: Run integration tests on real Docker services**
  Run:
  ```bash
  docker compose up -d postgres minio qdrant
  cd backend && pytest tests/integration/api/test_image_recognition_api.py -v
  ```
  Expected: PASS against real services.

- [ ] **Step 11: Lint and typecheck**
  Run:
  ```bash
  cd backend && ruff check . && ruff format --check . && mypy app
  ```
  Expected: clean.

- [ ] **Step 12: Commit**
  ```bash
  git add -A
  git commit -m "feat(phase1-sprint02): CPU ONNX image recognition API vertical slice"
  ```

---

## Task 2: Phase 1 Sprint 03 — Hardening, Logging, and Idempotency

**Files:**
- Create: `backend/app/domain/entities/process_event.py`
- Create: `backend/app/infrastructure/persistence/alembic/versions/0003_process_event_and_idempotency.py`
- Create: `backend/app/application/services/idempotency_service.py`
- Create: `backend/tests/integration/api/test_idempotency.py`
- Modify: `backend/app/api/main.py`
- Modify: `backend/app/api/routes/faces.py`

**Interfaces:**
- Consumes: existing endpoints and services.
- Produces: `Idempotency-Key` header handling; `process_event` persistence; structured validation errors.

- [ ] **Step 1: Add `process_event` table and repository**
  Migration must include `event_id`, `process_id`, `event_type`, `payload_json`, `created_at`.

- [ ] **Step 2: Implement idempotency gate**
  `idempotency_service.get_or_create(key, request_hint) -> existing process_id | None`.
  Routes read `Idempotency-Key` header and return cached result on duplicate.

- [ ] **Step 3: Add file size and format validation**
  Reject unsupported content types and oversized uploads with `ValidationError`.

- [ ] **Step 4: Tests and commit**
  Add `test_idempotency.py` and `test_validation_errors.py`; run full backend test suite; commit.

---

## Task 3: Phase 1 Sprint 04 — Docker Packaging and Acceptance

**Files:**
- Modify: `backend/Dockerfile`
- Create: `docker-compose.override.yml` (adds backend service)
- Modify: `Makefile`
- Create: `docs/implementation/review_packages/SPRINT-004-CODE-REVIEW-PACKAGE.md`

- [ ] **Step 1: Multi-stage backend Dockerfile**
  Base Python 3.12 slim, install system deps for opencv/headless, copy models from `backend/artifacts/models`, install wheel.

- [ ] **Step 2: Add backend service to compose**
  Healthcheck, depends_on for postgres/minio/qdrant, env_file `backend/.env`.

- [ ] **Step 3: Add acceptance target**
  ```makefile
  phase1-sprint-04-acceptance:
      docker compose build
      docker compose up -d
      cd backend && pytest tests/unit tests/integration -q
      cd backend && ruff check . && mypy app
      git diff --check
  ```

- [ ] **Step 4: Run acceptance and commit**
  Run `make phase1-sprint-04-acceptance` until exit 0, then commit and prepare review package.

---

## Task 4: Phase 2 Sprint 01 — Video Upload, Retention, and Async Job Foundation

**Files:**
- Create: `backend/app/domain/entities/video_asset.py`
- Create: `backend/app/domain/entities/video_job.py`
- Create: `backend/app/infrastructure/persistence/alembic/versions/0005_video_asset_and_job.py`
- Create: `backend/app/infrastructure/storage/video_asset_store.py`
- Create: `backend/app/application/services/video_upload_service.py`
- Create: `backend/app/application/services/video_job_worker_service.py`
- Create: `backend/app/api/routes/videos.py`
- Create: `backend/tests/integration/api/test_video_upload_and_job_state.py`

**Interfaces:**
- Consumes: MinIO adapter, UoW, UUID7 id generator.
- Produces:
  - `POST /videos/recognize` → `{ jobId, processId, status: "pending" }`
  - `GET /videos/jobs/{jobId}` → job status/progress
  - `DELETE /videos/jobs/{jobId}` → cancellation request

- [ ] **Step 1: Video asset and job domain models**
  States: `pending`, `processing`, `cancelling`, `completed`, `failed`, `cancelled`. Fields: lease_owner, lease_expires_at, cancellation_requested_at, attempt_no, heartbeat_at.

- [ ] **Step 2: Storage adapter for source videos**
  Upload to `mergenvision-source-videos` bucket with opaque key and SHA-256 verification; validate container/codec/duration/size before marking `ready`.

- [ ] **Step 3: Upload service**
  Accept multipart video, store temp, validate with FFmpeg probe, move to final MinIO object, create `VideoAsset` and `VideoJob` in PG, return job ID.

- [ ] **Step 4: Worker claim loop**
  `video_job_worker_service.claim_next()` uses `SELECT ... FOR UPDATE SKIP LOCKED`, sets lease + state=processing, returns job; no long-running DB transaction during GPU work.

- [ ] **Step 5: Cancellation protocol**
  `DELETE /videos/jobs/{jobId}` sets `cancellation_requested_at`; worker checks periodically; only transitions to `cancelled` after native process cleanup.

- [ ] **Step 6: Integration tests**
  Test upload → pending → claim → processing → completed/failed state machine on real services.

---

## Task 5: Phase 2 Sprint 02 — Native GPU Worker Integration

**Files:**
- Port from `MergenVisionPhase2/backend/native/` into a new top-level `native/` or `backend/native_worker/` folder.
- Create: `backend/app/infrastructure/native_worker/subprocess_adapter.py` (inspired by `MergenVisionPhase2/backend/app/infrastructure/native_worker/subprocess_adapter.py`)
- Create: `backend/app/ports/native_worker.py`
- Create: `backend/app/application/services/run_video_detection.py`
- Modify: `backend/app/application/services/video_job_worker_service.py`

**Interfaces:**
- Consumes: finalized MinIO video path, GPU device, job options.
- Produces: compact `observations.jsonl` with frame, pts_ns, original-resolution bbox, landmarks, detector score, 512-D embedding, model/profile identity.

- [ ] **Step 1: Port DeepStream 9.0 worker**
  Reuse `MergenVisionPhase2/backend/native/worker/main.cpp`, plugins (`gst-nvdsretinaface`, `gst-mvfacerecognizer`), kernels, recognition/gallery, and CMake build. Replace hardcoded `/app/backend/artifacts/engines` paths with env-driven config.

- [ ] **Step 2: Subprocess adapter**
  Run worker in Docker/subprocess, parse stdout `completed=...` summary and JSONL path, handle timeout/cancel.

- [ ] **Step 3: Wire worker into job lifecycle**
  `VideoJobWorkerService` calls `RunVideoDetectionService.execute(job)`; worker writes `observations.jsonl` to MinIO or shared path; result persisted on `completed`.

- [ ] **Step 4: Native build smoke**
  `make backend-native-build && make backend-native-smoke` exits 0 inside Docker.

---

## Task 6: Phase 2 Sprint 03 — Temporal Tracking and Identity Reconciliation

**Files:**
- Create: `backend/app/domain/video_tracking.py`
- Create: `backend/app/application/services/reconcile_video_identities.py`
- Create: `backend/app/infrastructure/tracking/byte_tracker_adapter.py`
- Create: `backend/tests/unit/tracking/test_reconcile_video_identities.py`

**Interfaces:**
- Consumes: `observations.jsonl`.
- Produces: `TrackletEvidence`, `CanonicalVideoPerson` with `face_id`, `status`, `tracklet_ids`, `appearances`, `best_shot`.

- [ ] **Step 1: Port ByteTrack/Kalman tracking core**
  Reference `MergenVisionPhase2/backend/native/tracking/byte_tracker.cpp`, `kalman_filter.cpp`, `multi_source_tracker.cpp`; reimplement pure-Python metadata tracker keyed by PTS/frame order.

- [ ] **Step 2: Build tracklet prototypes**
  Quality-filter observations, select temporally diverse embeddings, average and L2-normalize.

- [ ] **Step 3: Reconcile against PG/Qdrant gallery**
  Use `IdentityStorageLifecycleService` or direct Qdrant search + PG validation for known/anonymous decisions; cannot-link for overlapping same-source tracklets.

- [ ] **Step 4: Persist canonical persons and observations**
  New tables: `video_person`, `video_tracklet`, `video_appearance`, `face_observation` (or reuse `recognition_result`).

---

## Task 7: Phase 2 Sprint 04 — Aggregation and Overlay API

**Files:**
- Create: `backend/app/api/routes/video_results.py`
- Create: `backend/app/application/services/video_result_service.py`
- Create: `backend/app/api/contracts/video.py`
- Create: `backend/tests/integration/api/test_video_result.py`

- [ ] **Step 1: Aggregation service**
  Build person list with `firstSeen`, `lastSeen`, `totalDuration`, `appearances`, `detections` per `videorequirements.md` sections 4 and 11.

- [ ] **Step 2: Result endpoints**
  - `GET /videos/jobs/{jobId}/result`
  - `GET /faces/{faceId}/appearances`

- [ ] **Step 3: Overlay metadata endpoint**
  - `GET /videos/jobs/{jobId}/timeline?from=...&to=...` returns frame/PTS/bbox/name chunks for Canvas/SVG overlay.
  - Optional SSE/WebSocket push for progress and final overlay.

---

## Task 8: Phase 2 Sprint 05 — Internal UI and E2E Hardening

**Files:**
- Create/update frontend under `frontend/` (reuse patterns from `MergenVisionPhase2/frontend/` where applicable).
- Create: `frontend/src/api/timeline.ts`, `frontend/src/components/result/BboxCanvas.tsx`, etc.
- Create Playwright E2E tests.

- [ ] **Step 1: Backend-for-frontend proxy**
  Frontend talks only to versioned backend API; no direct MinIO/Qdrant access.

- [ ] **Step 2: Overlay player**
  Play original video with SVG/Canvas overlay using `requestVideoFrameCallback` metadata, `ResizeObserver`, DPR, fullscreen, seek.

- [ ] **Step 3: Rename after playback**
  Prove rename reflects immediately without re-encoding video.

- [ ] **Step 4: E2E, security, performance gates**
  Private buckets, short signed URLs, least-privilege credentials, input limits, benchmark report.

---

## Spec Coverage Check

| Requirement | Task |
|-------------|------|
| Image recognize/enroll/detail/delete/history/process endpoints | Task 1, 2 |
| Face detection/recognition status semantics | Task 1 |
| Persistent identity storage / multi-sample | Existing Sprint 01 + Task 1 |
| Process logging / history | Task 2 |
| Docker deployment | Task 3 |
| Video upload/retention | Task 4 |
| Async job state/cancel | Task 4 |
| Frame sampling / tracking | Task 5, 6 |
| Person aggregation / appearances | Task 6, 7 |
| Overlay metadata / UI | Task 7, 8 |

## Placeholder / Red-Flag Scan

- No "TBD"/"TODO" lines.
- No "implement later".
- Each task has concrete deliverables and verification commands.
- Later tasks depend on explicit interfaces created in earlier tasks.

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-07-17-mergenvision-phase1-phase2-roadmap.md`.**

Two execution options:

1. **Subagent-Driven (recommended):** dispatch a fresh subagent per task, review between tasks, fast iteration. Requires `superpowers:subagent-driven-development`.
2. **Inline Execution:** execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

**Which approach should we use?**
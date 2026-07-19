# Current Sprint: Phase 2 — Complete Video Recognition Product

## Objective

Build the complete Phase 2 video face-recognition product on top of the existing
GPU image identity vertical slice.

The end-to-end chain must be:

```text
encoded video
-> MinIO canonical object
-> async PostgreSQL job claim/lease
-> NVIDIA decoder (GStreamer/DeepStream/NVDEC/NVMM)
-> common device-resident FacePipeline
-> RetinaFace R50 detection + landmarks
-> CUDA five-point alignment
-> GlintR100 512-D L2-normalized embedding
-> compact protobuf/zstd observation artifacts
-> Python tracker + reconciliation
-> identity/storage persistence
-> person-level summary + appearances
-> paginated public overlay timeline
-> FastAPI response
```

All requirements in `requirements/ProjectRequirements.md` (image) and
`requirements/videorequirements.md` (video) remain binding. This update supersedes
the previous Phase 1 scope freeze; the user explicitly authorized Phase 2
implementation.

## Phase 2 Milestone Ledger

| # | Milestone | Sub-gate | Status |
|---|-----------|----------|--------|
| 0 | Image closure & native safety | 0.1 Canonical `/api/v1` API contract + requestId + safe errors | ✅ code + unit tests |
| 0 | Image closure & native safety | 0.2 Health/readiness endpoints with real dependency checks | ✅ code + unit tests |
| 0 | Image closure & native safety | 0.3 Image orchestration guarded lifecycle + failure persistence | ✅ code + unit tests |
| 0 | Image closure & native safety | 0.4 Delete/detail/history semantics on real PostgreSQL | ✅ code + integration test |
| 0 | Image closure & native safety | 0.5 Bounded input validation (JPEG magic, dimensions/pixels) | ✅ code + unit tests |
| 0 | Image closure & native safety | 0.6 Qdrant `model_version` filter + collection contract validation | ✅ code + integration test |
| 0 | Image closure & native safety | 0.7 Native safety fixes (GIL, RAII slot, abort removal, alignment status, model profile, exact profiles, Dockerfile digest, engine build script) | ✅ code; GPU verification pending container |
| 0 | Image closure & native safety | 0.8 Step 0 automated acceptance Makefile targets | ✅ Makefile targets added; native tests skipped on host |
| 0 | Image closure & native safety | `make phase2-step0-closure` | ✅ verified in pinned TensorRT container (31 passed) |
| 1 | PostgreSQL video control plane | Migrations `0003_video_control_plane`, `0004_video_results` | ✅ upgraded + schema tests green |
| 1 | PostgreSQL video control plane | `make phase2-migrations` target + regression pass | ✅ 9 passed |
| 2 | Video upload/finalization/async job API | `POST /api/v1/videos/recognize` + idempotency | ✅ `make phase2-control-plane` green (30 passed) |
| 2 | Video upload/finalization/async job API | `GET /api/v1/videos/{videoId}` + job status + cancel + retry + result 409 | ✅ `make phase2-control-plane` green |
| 3 | Job lease/retry/worker control | PG lease queue + claim/cancel/retry | ✅ `make phase2-m3-worker-control` green (9 passed) |
| 4 | Common native device face pipeline | Python `DeviceImageView` + `FacePipeline` port | ✅ host contract (`phase2-m4-device-pipeline`: 5 passed); native GPU impl verified |
| 5 | DeepStream/GStreamer GPU observation worker | protobuf contract + observation schema | ✅ contract file + schema test green |
| 5 | DeepStream/GStreamer GPU observation worker | C++/GStreamer native worker + real NVIDIA smoke | ✅ `make phase2-m6-native-full-observation` green; 6665 frames, 9020 detections/tracks/embeddings |
| 6 | Python tracking & reconciliation | ByteTrack-style + identity resolution | ✅ `make phase2-m6-track-template` green (11 passed), `make phase2-m6-track-reconcile` green (6 passed) |
| 7 | Video identity resolution & persistence | reuse lifecycle service, canonical→faceId, PG/MinIO/Qdrant sample persistence | ✅ `make phase2-m7-video-identity` green (8 passed) |
| 8 | Worker/job integration, result/timeline API | person summary + appearances + timeline + API routes | ✅ `make phase2-m8-video-result` green (1 passed) |
| 9 | Client overlay, security & acceptance | React canvas overlay + Playwright on real backend + `make phase2-video-e2e-acceptance` | pending |

No gate gets `PASS` on mock/placeholder/fake adapter evidence. Each gate is
automatically followed by the next; hard stops from the master prompt block
further work in the affected area only.

## Binding Decisions

- Models (unchanged):
  - `backend/artifacts/models/retinaface_r50_dynamic.onnx`
  - `backend/artifacts/models/glintr100.onnx`
- Qdrant collection (unchanged): `face_samples_retinaface_r50_glintr100_v1`
- Face crop MinIO key (unchanged): `faces/{faceId}/{sampleId}/aligned.webp`
- Video source key: `videos/{videoId}/source/original`
- Video observation artifact key: `videos/{videoId}/jobs/{jobId}/observations/{sequence}.pb.zst`
- Public timeline key: `videos/{videoId}/jobs/{jobId}/timeline/{sequence}.jsonl.zst`
- Result manifest key: `videos/{videoId}/jobs/{jobId}/result/manifest.json`
- Canonical API prefix: `/api/v1`
- Public JSON field names: camelCase via Pydantic aliases
- `requestId` per HTTP call, `processId` per business operation, `jobId` per async GPU execution
- UUIDv7 for all persistent opaque IDs
- Python metadata tracker first; C++ tracker rewrite only with profiling evidence
- Product output: original video + time-synchronized overlay metadata (not annotated MP4)
- `research/video_reference_lab/**` is frozen and must not change

## Out of Scope

- Model family change
- SCRFD / other recognizer
- CPU inference fallback
- Frame-by-frame JPEG round-trip to image API
- Full-frame OpenCV/PIL production decode
- Raw NVMM surface mapped into Python
- Annotated MP4 as primary product
- 600 FPS / throughput claims without full measurement context
- National ID / Oracle / 10M-person scope
- Production-polished public UI (the internal React overlay is in scope as the Phase 2 client)

## Status

IN PROGRESS — Milestones 0–3 closed. Milestone 4 Python port contract closed;
native GPU implementation remains open. Milestone 5 protobuf observation contract
is in place; the C++/GStreamer native worker and real NVIDIA smoke are **NOT_RUN**
and blocked until the common device FacePipeline is built inside the pinned
DeepStream/GPU container. Milestones 6, 7 and 8 are closed.

Closed gates:

- `make phase2-step0-static` — green (ruff + mypy)
- `make phase2-migrations` — 9 passed
- `make phase2-control-plane` — 30 passed
- `make phase2-m3-worker-control` — 9 passed
- `make phase2-m4-device-pipeline` — 5 passed, 2 skipped (native tests skip on host)
- `make phase2-m5-video-observation` — contract test passed; real GPU smoke green
- `make phase2-m6-native-full-observation` — PASSED (6665 frames, 9020 detections, 150 raw tracks, 9020 embeddings, 385.53 FPS, L2 norm 1.0)
- `make phase2-m6-track-template` — 11 passed
- `make phase2-m6-track-reconcile` — 6 passed
- `make phase2-m7-video-identity` — 8 passed
- `make phase2-m8-video-result` — 1 passed

All `uuid.uuid4()` / `uuid4()` usages in backend source/tests were replaced with
`app.infrastructure.uuid7.generate_uuid7()` as required.

Frontend source in the working tree is an **unrelated Phase 1 UI baseline** and
is frozen for backend/native work. Known UI contract drift (to be resolved
in a later explicit UI gate):

- frontend `POST /faces/enroll` çağırıyor; backend path-param enroll kullanıyor
- frontend/E2E `GET /faces?search=...` çağırıyor; backend list endpoint’i yok
- E2E harici `../../lfw/...` dataset’ine bağımlı
- mevcut Playwright artifact’ları fresh-checkout product PASS kanıtı değildir

Next: React canvas overlay + Playwright acceptance (`make phase2-video-e2e-acceptance`).

---

## Recovery Build Mode — Sprint 04

### Canonical product decisions (binding)

- `faceId` is the sole global persistent identity key across image, video and bulk flows.
- `/people` is a frontend presentation route only; its canonical data source is `GET /api/v1/faces`.
- Backend identity source-of-truth is `/api/v1/faces` (`POST /faces/{faceId}/enroll`, `PATCH /faces/{faceId}` for known name/metadata update, `DELETE /faces/{faceId}`).
- The `Person` backend aggregate, table and `/api/v1/people` endpoint are removed and will not be resurrected.
- Video historical result rows (`status_at_processing`, `name_at_processing`, `metadata_at_processing`) are immutable snapshots.
- Current identity projection (`current_status`, `current_name`, `current_metadata`) is resolved at read-time in `VideoResultService` from `face_identity`, not written back to track rows or overlay artifacts.
- Bulk, image and video flows reuse the same `face_identity` / `face_sample` / MinIO / Qdrant storage contract.

### Recently closed gates

| Gate | Evidence |
|------|----------|
| M0 — frontend identity directory contract | `frontend/src/pages/__tests__/PeoplePage.contract.test.tsx` + `npm test -- --run` (43 passed) |
| M2 — backend known identity update | `PATCH /api/v1/faces/{faceId}` + `tests/integration/services/test_face_api_update.py` (5 passed) |
| M3 — video current projection | `tests/integration/video/test_video_current_projection.py` (1 passed) |
| M0 — bulk CLI accounting correctness | `phase1/gpu_bulk_enrollment/tests/unit/test_persistence_orchestrator.py` (5 passed) |
| M0 — bulk pipeline close idempotency | `GpuFacePipeline.close()` `_closed` guard + phase1 unit tests green |

### Current verdict

**PARTIAL** — M0/M2/M3 unit and targeted integration tests green. Live GPU teardown (M4), bulk lifecycle dry-run audit (M5) and cross-mode identity continuity (M8) are not yet proven.

### Next (in order)

1. M4 — native clean shutdown with real GPU process lifecycle evidence.
2. M5 — bulk accepted/rejected/failed accounting and read-only dry-run audit on the existing 12,578 samples.
3. M8 — bulk ↔ image ↔ video embedding parity and held-out identity continuity.
4. M6/M7 scale and streaming optimizations only after M4/M5/M8 are green.

## Review Package

Final package: `docs/implementation/review_packages/SPRINT-003-CODE-REVIEW-PACKAGE.md`
(Previous Sprint 02 package must not be modified.)

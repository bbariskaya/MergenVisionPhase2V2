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
| 0 | Image closure & native safety | `make phase2-step0-closure` | pending real GPU container |
| 1 | PostgreSQL video control plane | Migrations `0003_video_control_plane`, `0004_video_results` | pending |
| 2 | Video upload/finalization/async job API | `POST /api/v1/videos/recognize` + idempotency | pending |
| 3 | Job lease/retry/worker control | PG lease queue + claim/cancel/retry | pending |
| 4 | Common native device face pipeline | `DeviceImageView` + shared `FacePipeline` | pending |
| 5 | DeepStream/GStreamer GPU observation worker | pinned container + NVDEC observation writer | pending |
| 6 | Python tracking & reconciliation | ByteTrack-style + identity resolution | pending |
| 7 | Result, timeline & appearance API | person summary + appearances + paginated timeline | pending |
| 8 | Retention, outbox & reconciliation worker | cleanup + failure recovery | pending |
| 9 | Docker compose, security & acceptance | `make phase2-acceptance` | pending |

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
- UI productization (existing UI changes in dirty tree are preserved but not a Phase 2 deliverable)

## Status

IN PROGRESS — Milestones 0.1–0.6 implemented with unit/integration tests.
M0.7 native safety code is complete: `ExecutionSlot` state enum, model profile
`crop_size` list parsing and dynamic profile validation, `ImageRuntime` dict
contract alignment, `backend/scripts/build_engines.py`, configurable CUDA
architectures, and `phase2-step0-*` Makefile targets. Static/type checks and
all non-native test suites pass; native GPU tests must run inside the pinned
TensorRT container because the host has no CUDA/TensorRT build environment.
M0.8 closure is pending that container run. Video control-plane work (M1–M9)
will follow after `make phase2-step0-closure` passes.

## Review Package

Final package: `docs/implementation/review_packages/SPRINT-003-CODE-REVIEW-PACKAGE.md`
(Previous Sprint 02 package must not be modified.)

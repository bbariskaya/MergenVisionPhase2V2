# Sprint 03 Code Review Package

## Scope

* Phase 2 Milestone 6 — Python tracking & reconciliation
* Phase 2 Milestone 7 — Video identity resolution & persistence
* Phase 2 Milestone 8 — Worker/job integration + result/appearance/timeline API

## Completion Verdict

Milestone 8: **PASS**

* The backend now drives a claimed `video_job` from compact observations through tracking → reconciliation → identity resolution → PostgreSQL persistence and exposes read-side people/appearances/timeline endpoints.
* `make phase2-m8-video-result` passes.
* ruff and mypy are clean.

The overall Sprint 03 remains **IN PROGRESS** because Milestone 9 (React canvas overlay + Playwright acceptance) has not started.

## Working User Behavior

1. Create a `video_asset` in `ready` state and a `processing` `video_job`.
2. Submit 5 synthetic `VideoObservationFrame`s to `VideoProcessingService.process(job_id, frames)`.
3. The job finishes with `state=completed`, `stage=finalize`, `person_count=1`, and a persisted `video_track` linked to a `recognition_result`.
4. Read the result via:
   * `GET /api/v1/videos/jobs/{job_id}/people`
   * `GET /api/v1/videos/jobs/{job_id}/appearances`
   * `GET /api/v1/videos/jobs/{job_id}/timeline`

## Validation Commands

```bash
make phase2-m8-video-result
# backend/tests/integration/video/test_video_processing_and_result_api.py::test_video_processing_completes_job_and_result_apis PASSED
# 1 passed

cd backend && .venv/bin/python -m ruff check app tests scripts
# All checks passed!

cd backend && .venv/bin/python -m mypy app
# Success: no issues found in 94 source files
```

## Raw Result Summary

| Gate | Result | Notes |
|------|--------|-------|
| `make phase2-step0-static` | green | ruff + mypy |
| `make phase2-migrations` | 9 passed | unchanged |
| `make phase2-control-plane` | 30 passed | unchanged |
| `make phase2-m3-worker-control` | 9 passed | unchanged |
| `make phase2-m4-device-pipeline` | 5 passed, 2 skipped | unchanged |
| `make phase2-m5-video-observation` | contract passed | unchanged |
| `make phase2-m6-native-full-observation` | PASSED | 6665 frames, 9020 detections, 150 raw tracks, 9020 embeddings, 385.53 FPS |
| `make phase2-m6-track-template` | 11 passed | unchanged |
| `make phase2-m6-track-reconcile` | 6 passed | unchanged |
| `make phase2-m7-video-identity` | 8 passed | unchanged |
| `make phase2-m8-video-result` | 1 passed | new |

## Changed Source Map

### New application ports

* `backend/app/application/ports/video_observations.py`
* `backend/app/application/ports/track_crop_provider.py`

### New domain entities

* `backend/app/domain/entities/video_track.py`
* `backend/app/domain/entities/video_tracking.py`

### New application services

* `backend/app/application/services/video_tracking_service.py`
* `backend/app/application/services/video_reconciliation_service.py`
* `backend/app/application/services/video_identity_resolution_service.py`
* `backend/app/application/services/video_track_persistence_service.py`
* `backend/app/application/services/video_processing_service.py`
* `backend/app/application/services/video_result_service.py`

### New infrastructure adapters

* `backend/app/infrastructure/runtime/track_crop_provider.py`

### Modified API layer

* `backend/app/api/routes/dependencies.py` — add `get_video_result_service`
* `backend/app/api/routes/videos.py` — add `/jobs/{job_id}/people`, `/appearances`, `/timeline`
* `backend/app/api/schemas.py` — add `VideoPeopleResponse`, `VideoPersonSummary`, `VideoAppearancesResponse`, `VideoAppearanceEntry`, `VideoTimelineResponse`, `VideoTimelineRecord`

### Modified persistence layer

* `backend/app/application/ports/repositories.py` — video repository contracts
* `backend/app/application/ports/unit_of_work.py` — video UoW + `flush`
* `backend/app/infrastructure/persistence/sqlalchemy/repositories/video_repositories.py` — implementation
* `backend/app/infrastructure/persistence/sqlalchemy/unit_of_work.py` — wiring

### New tests

* `backend/tests/unit/services/test_video_tracking_service.py`
* `backend/tests/unit/services/test_video_reconciliation_service.py`
* `backend/tests/unit/services/test_video_identity_resolution_service.py`
* `backend/tests/unit/services/test_video_track_persistence_service.py`
* `backend/tests/integration/video/test_video_identity_persistence.py`
* `backend/tests/integration/video/test_video_processing_and_result_api.py`

### Native / reference files (Phase 2 M5–M6)

* `backend/native/video_worker/CMakeLists.txt`
* `backend/native/video_worker/include/mv/video/recognition_mapper.hpp`
* `backend/native/video_worker/include/mv/video/video_face_pipeline.hpp`
* `backend/native/video_worker/src/video_face_pipeline.cpp`
* `backend/native/video_worker/src/kernels/warp_align_nv12_pitch.cu`
* `backend/native/video_worker/tests/real_batching_smoke.cpp`
* `backend/native/image_runtime/src/kernels/mergenvision_kernels.h`

### Build / project files

* `Makefile` — add `phase2-m8-video-result` target and include it in `phase2`
* `docs/implementation/CURRENT_SPRINT.md` — mark M8 complete; next gate M9
* `docs/implementation/IMPLEMENTATION_DETAILS.md` — add M8 section

### Unrelated or externally-updated

* `AGENTS.md` — updated outside of M8 scope (prompt-memory-mcp section).
* `decode_full.log` — runtime artifact, not source.
* `docs/superpowers/plans/2026-07-17-phase2-m6-m8-video-recognition.md` — planning artifact.

## Known Limitations

1. The native GPU worker does not yet materialize best-shot crops; the Python pipeline uses `PlaceholderTrackCropProvider` for new-anonymous track persistence.
2. The `result_manifest_key` is generated and persisted on the job, but the manifest JSON file is not yet written to MinIO.
3. Timeline endpoint returns all records (M8 scope); pagination will be added when needed for the client overlay.
4. M9 (React canvas overlay + Playwright acceptance) is the next gate and has not started.

## MCP / Skill Accountability

* `codebase-memory-mcp` — multi-file discovery across services, UoW, repositories, and API routes.
* `test-driven-development` — M8 started with a failing integration test that now passes.
* `verification-before-completion` — fresh verification run before claiming M8 passes.
* `using-superpowers` — active for this session.
* `context7`/Postman/playwright — not needed for this backend gate.

## Recommended Next Sprint

Milestone 9: React canvas overlay + Playwright end-to-end acceptance against real backend (`make phase2-video-e2e-acceptance`).

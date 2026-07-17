# MergenVision Phase 2 Build Notes

## Session recovery
- Repo: `/home/user/Workspace/MergenVisionPhase2v2`
- Branch: `main`
- Baseline: `c555c9a4e0982e77e63593b7de4bc4560715f612`
- HEAD: `38264357f5cce25a0b041b78e49cc683e04a6f9c` (worktree dirty due to M1/M2 work in progress)
- Build mode active, no subagents.

## Completed
### M0 — Image closure & native safety
- M0.1–M0.6 canonical API, health, image orchestration, delete/detail/history, JPEG validation, Qdrant model_version isolation.
- M0.7–M0.8 verified inside pinned `mergenvision-backend:gpu` TensorRT container:
  - `ExecutionSlot::State`, GIL release, RAII slot lease, structured native errors, model profile dynamic shapes, engine build script.
  - Full `make phase2-step0-closure` suite — **31 passed**.

### M1 — PostgreSQL video control plane
- Failing schema-contract tests added: `backend/tests/integration/persistence/test_phase2_migrations.py`.
- Migrations implemented:
  - `backend/app/infrastructure/persistence/alembic/versions/0003_video_control_plane.py`
  - `backend/app/infrastructure/persistence/alembic/versions/0004_video_results.py`
- `process_record` extended with `cancelled` status, `video_recognize` type, `cancelled_at`.
- Tables added: `video_asset`, `video_job`, `idempotency_record`, `process_event`, `outbox_event`, `video_track`, `video_tracklet`, `appearance_interval`, `video_timeline_chunk`, `video_track_sample`.
- `make phase2-migrations` target added; both old and new migration tests green (**9 passed**).

## Completed
### M2 — Video upload / finalization / async job API
- Domain models/repositories: `VideoAsset`, `VideoJob`, `IdempotencyRecord`
  (`app/domain/entities/video_*`, `app/infrastructure/persistence/sqlalchemy/repositories/video_repositories.py`).
- `VideoUploadService` + `VideoProbeService`: bounded streaming multipart upload,
  SHA-256, `ffprobe` subprocess validation, MinIO staging → canonical finalize,
  configurable limits and retention.
- API routes: `POST /api/v1/videos/recognize`, `GET /api/v1/videos/{videoId}`,
  `GET /api/v1/videos/jobs/{jobId}`, `DELETE /api/v1/videos/jobs/{jobId}`,
  `POST /api/v1/videos/jobs/{jobId}/retry`, `GET /api/v1/videos/jobs/{jobId}/result`.
- Integration tests (`tests/integration/video/test_upload_and_job.py`): 9 passed,
  covering happy path, missing idempotency key, idempotency replay, idempotency
  conflict, corrupt/unsupported video, cancel, retry, and premature result 409.

## Active
M3 — Job lease / retry / worker control:
- PostgreSQL lease queue (`FOR UPDATE SKIP LOCKED`), claim, heartbeat, expiry recovery.
- Worker scaffolding and cancellation propagation.

## Next concrete step
Add `JobLeaseService` with `claim_next_job`, `heartbeat`, `release_job`, and
expired-lease recovery, driven by failing integration tests against real PG.

## Environment
- Python: `backend/.venv/bin/python` (Python 3.12).
- Test services: `make phase2-services` uses `docker-compose.test.yml`.
- Native GPU work: pinned `mergenvision-backend:gpu` image.

## Next concrete step
Create `backend/app/domain/video_upload.py` service and `backend/app/api/v1/videos.py` router, then add integration tests for happy path and rejected video.

# GPU Bulk Enrollment + Video Identity Continuity — Build Mode Review Package

**Verdict:** `PARTIAL`

This package captures the Build Mode work done in session. Phase 2 schema contamination has been surgically removed and verified, and the isolated Phase 1 package has been corrected to use the real Phase 2 identity contract (no Person table, `aligned.jpg`, real crop bytes encoded with GPU nvImageCodec). The remaining runtime work (multi-GPU workers, real GPU/storage/video continuity acceptance) is documented as the next concrete gate.

---

## 1. Starting & ending HEAD

- Starting HEAD: `f81f9f446f3a538412fd007b89a3795ea3dceeb4`
- Ending HEAD: `f81f9f446f3a538412fd007b89a3795ea3dceeb4` (no commit made; corrections are in the working tree)
- Dirty worktree: Phase 2 files restored from `2cfde196` baseline, Person/bulk files deleted, Phase 1 files edited, untracked `promt15.txt`
- New protected-tree baseline generated: `.artifacts/phase1_gpu_bulk_enrollment/runs/649fa9f9-f688-4e37-b58e-7b393c64538d/baseline.json`

## 2. Database migration state

- Alembic current on persistent DB: `cf0441294c5f`
- Alembic head after removing `0006`: `cf0441294c5f`
- `0006_person_domain_and_identity_redirect.py` deleted
- Persistent DB facts (PostgreSQL localhost:5433/mergenvision):
  - `person` table: absent
  - `face_identity.person_id`, `face_identity.redirect_to_face_id`: absent
  - `process_record.process_type='face_assign'`: 0 rows
  - active known `face_identity`: 2
  - total `face_identity`: 246
  - total `face_sample`: 249
- Conclusion: **Branch A** — 0006 never applied to the persistent DB, so surgical source correction is safe.

## 3. 0a2 contamination exact map

Changes introduced by `0a2d928` and retained until this session:

- Person domain entity/ORM/repository/service/API routes
- `face_identity.person_id` and `redirect_to_face_id`
- `PersonManagementService`, bulk enrollment controller/service
- `/people/*` routes, frontend PeoplePage
- Migration `0006`
- Person-coupled tests

All of the above have been removed or reverted. Independent fixes preserved:

- `ProcessRecord.fail/cancel` no longer sets `completed_at`
- `video_worker_main.py` `VideoObservationFrame` import restored

## 4. Phase 2 correction verification

| Test suite | Result |
|------------|--------|
| `tests/unit` | 170 passed |
| `tests/integration/persistence/test_migrations.py` + `test_phase2_migrations.py` | 9 passed |
| `tests/integration` (full) | 62 passed, 1 failed (environmental Docker port conflict on restart test) |
| `ruff check app` | passed |
| `protected-tree gate` | PASS |

The failing integration test is `test_data_survives_restart` failing because Docker Compose could not bind `127.0.0.1:59000` (already allocated). It is not a source regression.

## 5. Phase 1 isolated package corrections

Files changed under `phase1/gpu_bulk_enrollment/`:

- `python/mv_phase1_bulk/types.py` — removed `PersonRecord`, `person_id`, `EnrollmentBundle.person_id`, `EnrollmentOutcome.person_id`
- `python/mv_phase1_bulk/ids.py` — removed `make_person_id`, `PersonId`; object key now `faces/{face_id}/{sample_id}/aligned.jpg`
- `python/mv_phase1_bulk/identities.py` — `SubjectBundle` no longer carries a Person
- `python/mv_phase1_bulk/postgres_store.py` — no `person` table reads/writes; writes only `face_identity` + `face_sample`
- `python/mv_phase1_bulk/persistence.py` — persists aligned JPEG crop bytes instead of original JPEG
- `python/mv_phase1_bulk/minio_store.py` — object key/content type aligned with `aligned.jpg` / `image/jpeg`
- `python/mv_phase1_bulk/pipeline.py` — converts aligned chips to uint8 HWC on GPU and encodes `image/jpeg` with nvImageCodec; no PIL/CPU encode
- `python/mv_phase1_bulk/cli.py` — `EnrollmentOutcome` no longer references person
- `pyproject.toml` — removed `pillow` dependency, added `nvidia-nvimgcodec-cu12` GPU JPEG encoder
- `tests/unit/test_ids.py`, `tests/unit/test_persistence_orchestrator.py` — updated for Person-free contract

| Test suite | Result |
|------------|--------|
| `phase1/gpu_bulk_enrollment/tests/unit` | 34 passed, 1 skipped |
| `ruff check python/mv_phase1_bulk tests/unit` | 1 pre-existing UP038 only; no new issues |

## 6. Model / engine contract

- ONNX paths and SHAs from user prompt are preserved as the target contract.
- `phase1/gpu_bulk_enrollment/config/model_profile.json` still contains absolute paths and a not-yet-built `artifacts/engines/default/` engine layout.
- Existing built engines are at `backend/artifacts/engines/`:
  - `retinaface_r50_dynamic.bs1.opt8.max64.fp16.trt1016.engine`
  - `glintr100.bs1.opt8.max64.fp16.trt1016.engine`
- Native CUDA extension (`_mv_phase1_bulk_native`) has **not** been built yet.

## 7. Blockers / NOT_RUN gates

The following gates are intentionally not claimed as PASS because they require native build/runtime and GPU resources not exercised in this session:

1. Native CUDA extension build (`pip install -e phase1/gpu_bulk_enrollment` with scikit-build-core + CMake + CUDA)
2. TensorRT engine build or adapter to the existing `backend/artifacts/engines/*.trt1016.engine` files
3. Real GPU batch extraction (batch 1 vs N parity, detector score sync, aligned WebP decode check)
4. Real cross-subject microbatching (currently CLI still loops subject-by-subject)
5. Real multi-GPU worker/sharding (`--gpu-devices` currently only uses the first device)
6. Streaming manifest with bounded producer/consumer queues
7. Real MinIO/Qdrant/PostgreSQL integrated storage smoke test with actual aligned WebP artifacts
8. Real video upload/job/run and `known` continuity with the same `face_id`
9. Frontend PeoplePage/API cleanup (source files still exist but backend endpoints are gone)

## 8. Reference / provenance

- Protected Phase 2 baseline source: `2cfde196795a2b783e4494d879bbc48fe3361f69`
- Read-only GPU/orchestration reference: `https://github.com/bbariskaya/MergenVisionDemo` commit `5bf4b4c57542b26058e8d068186faee06c0fc29c` (not yet adapted in code due to native build blocker)
- `21st` and `Ruflo` were not used.

## 9. MCP / skill accountability

- `prompt-memory-mcp`: used for gate notes and recall
- `codebase-memory-mcp`: repository indexed (5396 nodes, 23838 edges) for call-graph discovery
- `systematic-debugging`: applied to classify contamination before fixing
- `test-driven-development`: existing tests extended/fixed before code changes
- `context7` / `deepwiki`: skipped; no version-specific library questions required during this batch

## 10. Recommended next sprint

**Build the native Phase 1 runtime and run a real GPU/storage acceptance test.**

Concrete first step: build `_mv_phase1_bulk_native` against the existing TensorRT/CUDA libraries and the existing `backend/artifacts/engines/*.trt1016.engine` files, then run the small deterministic subset (3 subjects, 2 images each, 1 no-face, 1 multi-face, 1 corrupt JPEG) end-to-end through `mv-phase1-bulk enroll` and verify PostgreSQL/MinIO/Qdrant state.

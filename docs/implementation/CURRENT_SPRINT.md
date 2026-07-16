# Current Sprint: Phase 1 Sprint 01 — Minimal Identity Storage Foundation

## Objective

Establish real PostgreSQL, MinIO and Qdrant connections and prove the persistent identity lifecycle using a deterministic test vector and a tiny valid image fixture.

## In Scope

- Four PostgreSQL tables: `face_identity`, `face_sample`, `process_record`, `recognition_result`.
- One initial Alembic migration (`0001_identity_storage_foundation.py`).
- Forward-only correctness migration (`0002_sprint01_correctness.py`).
- Pure Python domain layer with explicit state transitions.
- SQLAlchemy 2.0 async repository adapters and unit of work.
- MinIO adapter for the `mergenvision-face-samples` bucket.
- Qdrant adapter for the `face_samples_v1` cosine 512-D collection.
- `IdentityStorageLifecycleService`:
  - `resolve_or_create`
  - `add_sample`
  - `enroll_identity`
  - `deactivate_identity`
- Integration tests on real Docker services.

## Out of Scope

- API endpoints / FastAPI
- React UI
- Real detection / recognition / alignment / GPU inference
- Video / tracking
- `inference_profile`, `idempotency_record`, `face_observation`, `process_event`, `outbox_event`
- Outbox, saga, reconciliation, dead-letter
- Model artifact SHA, requirement SHA checks
- National-ID, Oracle, 10M-person scope

## Correction Base

Reviewed base commit: `675549670ab65daec9ffedae3e62ddb4f4478dc3`.

## Status

`COMPLETED_PENDING_SENIOR_REVIEW`.

All mandatory local correction gates passed twice on 2026-07-16 (77 tests, static analysis, Docker build/import, restart probe, git diff --check). This sprint is **not** full Phase 1 completion. Sprint 02 has not been started.

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

All mandatory local correction gates passed twice on 2026-07-16 (78 tests, static analysis, Docker build/import, restart probe, git diff --check). This sprint is **not** full Phase 1 completion. Sprint 02 has not been started.

## Isolated Technical Spike Exception

User-approved isolated video reference correctness spike. This does not begin product Sprint 02, does not alter the mandatory product implementation order, and cannot claim production video/GPU completion.

### Video Reference Lab — Forensic Correction (2026-07-16)

Corrections applied to `research/video_reference_lab/`:

- `evaluation.py` now compares observations through the
  `observation_id -> raw_tracklet_id -> canonical_track_id` mapping instead of
  mixing namespaces.
- `evaluate_gallery` counts a track as `known` only when
  `decision_reason == "gallery_match"` and `display_label is not None`.
- `tests/unit/test_evaluation.py` rewritten to validate the corrected semantics.
- `byte_tracker.py` moved raw-tracklet ID allocation to a per-tracker counter
  (`_allocate_tracklet_id`) so repeated tracker instances are deterministic and
  independent.
- `benchmark.py` now instruments `active_tracklet_ids()` each frame and reports
  a real `max_active_tracks_estimate` instead of always returning `0`.

Verification:

- `make video-reference-unit` — 165 passed, 1 skipped, 1 xfailed.
- `make video-reference-static` — ruff + mypy + format clean.
- `make video-reference-synthetic-e2e video-reference-artifact-integrity video-reference-chunk-parity` — all passed.
- Full `run-friends --max-frames 300` completed end-to-end.
  - 300 frames, 1,792 observations, 265 valid embeddings, 537 tracking-eligible assignments.
  - 13 raw tracklets -> 8 valid templates -> 10 canonical tracks (3 merges).
  - Chunk invariance verified across chunk sizes [1, 8, 17, 64].
  - Replay benchmark: ~2,500 FPS, `max_active_tracks_estimate` = 5.
  - Gallery decision: 2 known (Rachel, Monica), 8 unknown. No ground truth is
    available, so identity accuracy is **not proven**; only structural correctness
    and safety rules are demonstrated.

Caveats honestly reported:

- `ground_truth_available: false` for `Friends.mp4`; pairwise identity precision/recall cannot be computed.
- 5 of 13 raw tracklets have zero recognition-eligible observations, so they
  cannot contribute to gallery decisions.
- 3 gallery decisions are `unknown` because top-1 cosine is below the calibrated
  `match_threshold` (0.45). These are not forced to `known`.

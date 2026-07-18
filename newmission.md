# MergenVision Phase 2+ Mission Plan

> Long-term objective from `prompt12.txt`: unify global identity/enrollment, add bulk dataset enrollment, and build an isolated GPU video research lab.
> Plan status: **deep discovery complete; ready for implementation**.
> Last updated: 2026-07-18.

---

## 0. Current Repository Snapshot

| Area | State |
|------|-------|
| Branch | `main` |
| Backend | FastAPI + SQLAlchemy 2.0 + Alembic + PostgreSQL/MinIO/Qdrant |
| Frontend | React + Vite + TanStack Query (unrelated Phase 1 baseline, being updated) |
| Native | C++/CUDA/TensorRT GStreamer video worker exists, pinned DeepStream 9 container |
| Tests | Backend pytest suites green through M8; frontend/Playwright pending |
| Existing video lab | `research/video_reference_lab/**` is **FROZEN** and must not change |
| Person domain | **Does not exist**; only `face_identity` exists |
| Bulk enrollment | **Does not exist**; only unused `bulkJob` query keys in frontend |
| GPU video lab | `research/gpu_video_lab/` **does not exist** yet |

### Critical findings from deep discovery

1. **Stale video UI after enrollment** — root cause is `VideoResultService.list_people` returning `track.status_at_processing`/`name_at_processing` (`backend/app/application/services/video_result_service.py:68-69`). `VideoPersonSummary` schema has no `currentStatus`/`currentName` fields. Frontend `TrackListPanel` renders those stale fields.
2. **No Person domain** — `face_identity` is the only identity aggregate. `person` table/entity/relationship is required by `architectureplan.md` and `AGENTS.md` §17 but missing.
3. **Enrollment is promotion-only** — `IdentityStorageLifecycleService.enroll_identity` only promotes `anonymous -> known` for the same `face_id`. There is no "assign to existing person" / merge / redirect flow.
4. **Frontend error parser gap** — `frontend/src/api/client.ts:parseError` reads only `body.detail`, ignores backend's structured `body.error.message`/`body.error.code` envelope (`backend/app/api/schemas.py:25-34`).
5. **Missing query invalidation** — `useEnrollMutation` invalidates `face`, `faceHistory`, `faces` but never `video-people`, so the sidebar stays stale.
6. **Bulk enrollment is absent** — no backend service/route/worker; only unused frontend query keys.
7. **Makefile gaps** — no `global-identity-enrollment-acceptance`, `bulk-enrollment-e2e`, or `gpu-video-lab-*` targets.

---

## 1. Implementation Order (Hard Stop: No Skipping)

Follow `AGENTS.md` §4 mandatory order:

1. Requirement / contract / ERD / state-machine freeze for all three milestones.
2. PostgreSQL migration + model + domain entity changes (Person, identity redirect, current projection).
3. Backend API contract updates and read-side projection (`VideoPersonSummary` current fields).
4. Enrollment unification backend (new person + assign/merge to existing person).
5. Frontend type/parser/invalidation fixes + enroll UI for new/existing person.
6. Backend bulk enrollment domain/API/worker.
7. Frontend bulk enrollment status UI.
8. Isolated GPU video lab skeleton (capture/replay/sweep).
9. Makefile acceptance targets + integration tests for all three milestones.
10. Full E2E acceptance and review package.

---

## 2. Milestone A — Global Identity & Enrollment Unification

### 2.1 Goal

- Every known `face_identity` belongs to a `person`.
- Video results show both the immutable job-time decision (`statusAtProcessing`, `nameAtProcessing`) and the mutable current projection (`currentStatus`, `currentName`).
- Enrollment supports two explicit modes: **create new person** and **assign to existing person** (merge/redirect).
- Frontend displays real errors and refreshes video people after enrollment.

### 2.2 ERD Changes

```text
person
  person_id UUID PK
  display_name text not null
  person_metadata jsonb not null default '{}'
  is_active bool not null default true
  created_at timestamptz
  updated_at timestamptz
  deleted_at timestamptz

face_identity
  + person_id UUID FK -> person.person_id nullable initially
  + redirect_to_face_id UUID FK -> face_identity.face_id nullable
  + merge_confidence float nullable
  + merge_process_id UUID FK -> process_record.process_id nullable
  status check ('anonymous', 'known')
  (add: known identity must have person_id not null)

face_sample (unchanged)
recognition_result (unchanged)
video_track (unchanged — keeps face_id as historical link)
```

Backfill migration: create one `person` per existing known `face_identity`, copy `display_name` and `identity_metadata` into `person.person_metadata`, set `face_identity.person_id`. Existing anonymous identities keep `person_id = NULL` until enrolled.

### 2.3 Domain Changes

- New `Person` aggregate in `backend/app/domain/entities/person.py`.
- `FaceIdentity` gains `person_id: PersonId | None` and `redirect_to_face_id: FaceId | None`.
  - `redirect_to_face_id` makes the old face_id an alias; API lookups follow the redirect to the canonical face_id.
- `FaceIdentity.promote_to_known(display_name, metadata)` becomes `promote_to_known(person_id, metadata)` OR keep `display_name` and create a person internally; explicit mode is cleaner.
- New `FaceIdentity.assign_to_person(person_id)` for merging an anonymous/known identity into an existing person; marks source as `redirect_to_face_id = canonical face_id` and deactivates source samples in Qdrant.
- Add `IdentityMergeAudit` event / process_record detail for traceability.

### 2.4 Backend API Contract

#### `POST /api/v1/faces/{face_id}/enroll`

Request body (`EnrollByFaceIdRequest`):

```json
{
  "mode": "new_person" | "existing_person",
  "name": "Rachel Green",
  "metadata": { "department": "IT" },
  "targetPersonId": "uuid"   // required when mode == existing_person
}
```

- `new_person`: create `person`, link `face_identity` to it, promote to `known`. Return `EnrollResponse` with `person_id`.
- `existing_person`: validate `target_person_id` exists and is active, mark `face_identity.redirect_to_face_id = canonical face_id of that person`, deactivate source identity and samples, add merge audit. Return canonical `face_id` and `person_id`.

Add validation errors:
- `CANNOT_ASSIGN_TO_SELF`
- `TARGET_PERSON_NOT_FOUND`
- `SOURCE_IDENTITY_NOT_ACTIVE`
- `ALREADY_KNOWN_DIFFERENT_PERSON` (conflict)

#### `GET /api/v1/people`

List/search persons; used by frontend enroll UI dropdown.

#### `GET /api/v1/people/{person_id}`

Person detail with all linked `face_identity` rows (including redirects).

#### `GET /api/v1/people/{person_id}/appearances`

Cross-video appearances aggregated by person (future enhancement; start with face_id-based).

### 2.5 Read-Side Projection for Video Results

Update `backend/app/api/schemas.py`:

```python
class VideoPersonSummary(_PublicBaseModel):
    track_id: str
    face_id: str
    person_id: str | None = None
    status_at_processing: str
    name_at_processing: str | None = None
    current_status: str
    current_name: str | None = None
    identity_version_at_processing: int
    current_identity_version: int
    ...
```

Update `VideoResultService.list_people` to:
1. Load tracks for job.
2. Collect distinct `face_id`s.
3. Load current `face_identity` + `person` rows in one query (with redirect resolution).
4. Build summary with both historical and current fields.

Update `backend/app/api/routes/videos.py:_to_person_summary` accordingly.

### 2.6 Frontend Changes

#### `frontend/src/api/types.ts`

```typescript
export interface VideoPersonSummary {
  track_id: UUID
  face_id: UUID
  person_id: UUID | null
  status_at_processing: RecognitionStatus
  name_at_processing: string | null
  current_status: RecognitionStatus
  current_name: string | null
  ...
}
```

Update `ApiErrorResponse`:

```typescript
export interface ApiErrorBody {
  code: string
  message: string
  retryable?: boolean
  details?: Record<string, unknown>
}

export interface ApiErrorResponse {
  error?: ApiErrorBody
  detail?: string | ApiErrorDetail[]
}
```

#### `frontend/src/api/client.ts`

Update `parseError`:

```typescript
const message =
  body?.error?.message ??
  (typeof body?.detail === 'string' ? body.detail : `${response.status} ${response.statusText}`)
const code = body?.error?.code ?? `HTTP_${response.status}`
```

#### `frontend/src/api/faces.ts`

Update `useEnrollMutation.onSuccess` to also invalidate `queryKeys.videoPeople` for every affected job. Since the mutation does not know job IDs, invalidate all `video-people` queries conservatively:

```typescript
queryClient.invalidateQueries({ queryKey: ['video-people'] })
queryClient.invalidateQueries({ queryKey: ['video-result'] })
```

#### `frontend/src/pages/EnrollPage.tsx`

- Add mode selector: "Yeni kişi oluştur" / "Mevcut kişiye ata".
- In "existing person" mode, show searchable dropdown populated by `usePeople` hook (new).
- On success, display canonical `face_id`, `person_id`, and current name.

#### `frontend/src/components/video/TrackListPanel.tsx`

- Show `current_name` primary, `name_at_processing` as secondary/subtle label if different.
- Badge from `current_status`.
- "Adlandır" link uses `face_id` (following redirect is backend responsibility).
- Add tooltip/hint when identity was merged/reassigned after processing.

### 2.7 Tests

- Unit: `test_person_domain.py`, `test_face_identity_redirect.py`.
- Integration: migration backfill, enrollment new/existing person, merge redirect, stale UI projection.
- API: `test_face_enroll_unified.py`, `test_video_people_current_projection.py`.
- Frontend: enroll page mode switch, error parser, query invalidation.

### 2.8 Acceptance Target

```makefile
global-identity-enrollment-acceptance:
	# runs backend integration tests + frontend typecheck + Playwright enroll/rename flow
```

---

## 3. Milestone B — Bulk Dataset Enrollment

### 3.1 Goal

Import the user's local celebrity/identity photo dataset into the gallery using the same GPU models and lifecycle as single-image recognition, with manifest, admission/quality gates, and idempotent retry.

### 3.2 Domain

New aggregate `BulkEnrollmentJob`:

```text
bulk_enrollment_job
  job_id UUID PK
  manifest_source_path text
  manifest_sha256 char(64)
  state check ('pending','processing','completed','failed','cancelled')
  total_entries int
  processed_entries int
  admitted_entries int
  failed_entries int
  error_summary jsonb
  options jsonb  # quality thresholds, duplicate policy, etc.
  lease_owner / lease_token / lease_expires_at (if async worker)
  created_at / updated_at / completed_at
```

New child table `bulk_enrollment_entry` (one row per source file, optional many faces per file):

```text
bulk_enrollment_entry
  entry_id UUID PK
  job_id UUID FK
  source_path text not null
  source_sha256 char(64)
  inferred_person_name text
  state check ('pending','admitted','rejected','failed')
  rejection_code text
  face_id UUID FK nullable
  sample_id UUID FK nullable
  person_id UUID FK nullable
  detection_count int
  processed_at timestamptz
```

### 3.3 Manifest Format

Accepted manifest inputs:

1. **Folder convention** (default): `dataset/{person_name}/{image1.jpg, image2.jpg, ...}`.
2. **JSON manifest**: `{ "entries": [ { "path": "...", "personName": "..." } ] }`.

Manifest is generated or validated by a CLI/admin step and persisted to MinIO with SHA-256.

### 3.4 Admission / Quality Gates

For each image:

- Decode probe (reject `DECODE_FAILED`).
- Exactly one face detected (reject `NO_FACE`, `MULTIPLE_FACES`).
- Minimum face size in pixels (configurable).
- Quality score >= threshold (blur/pose/landmark geometry).
- Alignment residual <= threshold.
- Deduplication: if SHA-256 already admitted in this or a previous job, skip idempotently.
- Optional: compare to existing gallery; if top-1 match >= strict threshold and top-2 margin ok, assign to existing person instead of creating new.

### 3.5 Processing Flow

1. API creates `BulkEnrollmentJob` in `pending`, uploads/validates manifest, returns `job_id`.
2. Worker picks job via `FOR UPDATE SKIP LOCKED` lease.
3. For each entry:
   - Read local file bytes (worker must have access to the dataset path; no browser upload).
   - Run existing `ImageRecognitionService.recognize_image` or internal `FacePipeline` directly.
   - Apply gates.
   - On admit: reuse existing identity if strong match and policy allows; otherwise create new `face_identity` + `person` + `face_sample` through `IdentityStorageLifecycleService`.
   - Update `bulk_enrollment_entry` state.
4. Job completes; `error_summary` contains per-code counts.

### 3.6 Idempotency & Retry

- Idempotency key = manifest source path + source SHA-256 + job options hash.
- `bulk_enrollment_entry` unique on `(job_id, source_sha256)`.
- Retry endpoint re-runs failed/pending entries only; admitted entries are skipped.
- Duplicate across jobs: if source SHA-256 already linked to an active sample, mark `DUPLICATE_ALREADY_ADMITTED`.

### 3.7 Backend API

- `POST /api/v1/bulk-enrollment/jobs` — create from manifest path/options.
- `GET /api/v1/bulk-enrollment/jobs/{job_id}` — status + counts.
- `GET /api/v1/bulk-enrollment/jobs/{job_id}/entries` — paginated entry list.
- `POST /api/v1/bulk-enrollment/jobs/{job_id}/retry` — retry failed entries.
- `DELETE /api/v1/bulk-enrollment/jobs/{job_id}` — cancel.

### 3.8 Frontend

- `frontend/src/api/bulkEnrollment.ts` with hooks: `useCreateBulkEnrollmentMutation`, `useBulkJob`, `useBulkJobEntries`.
- Page `frontend/src/pages/BulkEnrollmentPage.tsx` to submit local manifest path, view progress, download error report.
- Re-use `queryKeys.bulkJob`/`latestBulkJob`.

### 3.9 Tests

- Unit: admission gate logic, manifest parsing.
- Integration: create job, process entries, idempotency, retry, gallery growth, duplicate rejection.
- Fixture: small synthetic dataset in `backend/tests/fixtures/bulk_dataset/`.

### 3.10 Acceptance Target

```makefile
bulk-enrollment-e2e:
	# validates manifest -> job -> worker -> gallery entries -> idempotent retry
```

---

## 4. Milestone C — Isolated GPU Video Lab

### 4.1 Goal

Build `research/gpu_video_lab/`, completely separate from the frozen `research/video_reference_lab/`, to capture/replay/sweep tracker, quality, best-shot, and reconciliation variants against the production GPU observation pipeline, with promotion candidate / shadow mode output.

### 4.2 Directory Layout

```text
research/gpu_video_lab/
  pyproject.toml
  src/mv_gpu_lab/
    __init__.py
    cli.py
    capture/          # run native worker, save observations
    replay/           # feed observations to Python pipeline
    sweep/            # parameter grid
    evaluate/         # metrics + promotion candidate
    storage.py        # artifact/MinIO helpers
    models.py         # lab-specific data classes
  configs/
    friends_capture.yaml
    friends_sweep.yaml
  tests/
    unit/
    integration/
```

### 4.3 Capture

- Run the existing native worker (`backend/native/video_worker/build/real_batching_smoke` or production worker binary) on a video.
- Save compact observation artifacts to `research/gpu_video_lab/data/{video_id}/observations/{sequence}.pb.zst` (same protobuf contract as production).
- Save metadata: model/engine SHA, batch sizes, hardware UUID, duration, FPS.

### 4.4 Replay

- Load captured observations without re-running GPU.
- Run `VideoTrackingService`, `VideoReconciliationService`, `VideoIdentityResolutionService`, `VideoTrackPersistenceService`, and `VideoOverlayService` exactly as production.
- Emit people/appearances/timeline for evaluation.

### 4.5 Sweep

- Grid over:
  - tracker max lost age / IoU threshold
  - reconciliation overlap threshold
  - recognition match threshold / margin multiplier
  - max template samples / min template gap
  - best-shot quality gate
- Each config runs replay and writes metrics.
- Aggregate results and pick promotion candidate (best F1 / IDF1 / recall trade-off).

### 4.6 Shadow Mode

- Promotion candidate config can be marked `shadow: true`.
- Re-runs latest completed production job's observations and writes a shadow result manifest next to the real one; does not mutate production DB.
- Diff report against production result.

### 4.7 Makefile Targets

```makefile
PHONY += gpu-video-lab-capture gpu-video-lab-replay gpu-video-lab-sweep gpu-video-lab-acceptance

gpu-video-lab-capture:
	cd research/gpu_video_lab && $(LAB_PYTHON) -m mv_gpu_lab.cli capture --config configs/friends_capture.yaml

gpu-video-lab-replay:
	cd research/gpu_video_lab && $(LAB_PYTHON) -m mv_gpu_lab.cli replay --config configs/friends_capture.yaml

gpu-video-lab-sweep:
	cd research/gpu_video_lab && $(LAB_PYTHON) -m mv_gpu_lab.cli sweep --config configs/friends_sweep.yaml

gpu-video-lab-acceptance: gpu-video-lab-capture gpu-video-lab-replay gpu-video-lab-sweep
	cd research/gpu_video_lab && $(LAB_PYTHON) -m mv_gpu_lab.cli evaluate --config configs/friends_sweep.yaml
```

### 4.8 Tests

- Unit: sweep grid generation, metric computation.
- Integration: capture->replay round-trip yields same people count, sweep finds a promotion candidate.

---

## 5. Cross-Milestone Concerns

### 5.1 Migration Strategy

- Add a single new Alembic migration `0006_global_identity_and_person_domain.py`.
- Do **not** modify existing migrations `0001`–`0005` or the fix migration.
- Backfill existing known identities into `person` rows; leave anonymous `person_id = NULL`.
- Add `face_identity.redirect_to_face_id` nullable FK.

### 5.2 UUIDv7

All new persistent IDs (`person_id`, `bulk_enrollment_job.job_id`, etc.) use `app.infrastructure.uuid7.generate_uuid7()`.

### 5.3 Object Keys

No PII in MinIO keys. Person name must not appear in object keys, Qdrant payloads, or raw logs.

### 5.4 Qdrant

- Qdrant still indexes only `face_sample.sample_id` vectors with technical payload (`face_id`, `sample_id`, active flag, model version).
- Person name/metadata must never be written to Qdrant.

### 5.5 Test-Driven Discipline

Per `AGENTS.md` §31: failing test → minimum implementation → unit test → integration test → real-service smoke → lint/type/build → diff review.

### 5.6 Completion Verdicts

Only `PASS`, `PARTIAL`, `BLOCKED`, `NOT_TESTED`.

---

## 6. Immediate Next Steps (First Sprint)

1. **Contract freeze** — this plan is the contract; open questions below.
2. **Migration `0006`** — create `person`, add `face_identity.person_id` + `redirect_to_face_id`, backfill.
3. **Domain entity `Person`** + `FaceIdentity` redirect support.
4. **Update `VideoPersonSummary` schema and `VideoResultService.list_people`** to project current identity.
5. **Update frontend types and `TrackListPanel`** to render current name/status.
6. **Fix `client.ts` error parser**.
7. **Add `useEnrollMutation` video-people invalidation**.
8. **Implement `POST /faces/{face_id}/enroll` new-person mode**.
9. **Implement `POST /faces/{face_id}/assign` (or mode parameter) for existing-person merge**.
10. **Update `EnrollPage` with mode selector and person search**.
11. **Add `global-identity-enrollment-acceptance` Makefile target** and make it green.

---

## 7. Binding Decisions (Confirmed 2026-07-18)

1. **Assign-to-existing-person / merge semantics — REDIRECT/ALIAS.**
   - Source `face_id=A` becomes inactive and carries `canonical_face_id=B` pointer.
   - `/faces/A` API calls continue to work and resolve to canonical identity `B`.
   - New recognition and current projection use `B`.
   - Historical recognition/video snapshots remain immutable as `A`.
   - Video UI shows `B`'s current name; UI route opened with `A` may navigate to canonical `B`.

2. **Bulk dataset ingestion — READ-ONLY BIND MOUNT.**
   - A dedicated bulk-enrollment worker container reads the dataset from a user-supplied read-only bind mount.
   - The normal backend container does not access the dataset path.
   - Dataset local path lives only in worker config; it must not leak to API responses, logs, or MinIO object keys.
   - Worker writes accepted aligned crops to MinIO; PostgreSQL stores manifest/provenance/SHA; Qdrant receives only accepted embeddings.
   - A MinIO-source mode may be added later for distributed/remote import.

3. **GPU video lab — CAPTURE + REPLAY/SWEEP.**
   - **Capture:** real GPU hot path inside pinned `mergenvision/deepstream-dev:9.0` produces an immutable observation bundle.
   - **Replay/sweep:** CPU-fast tracker/quality/template/reconciliation experiments over the captured bundle.
   - Core architecture: GPU capture once → CPU replay many → finalist GPU rerun.

---

## 8. Risk Register

| Risk | Mitigation |
|------|------------|
| Modifying frozen migrations | Add new migration only; verify `alembic upgrade head` from clean and existing DB. |
| Redirect lookup N+1 in video results | Load all current identities in one query in `VideoResultService`. |
| Merge breaks existing URLs/bookmarks | Source `face_id` stays valid via redirect; API resolves it transparently. |
| Bulk enrollment gallery poisoning | Strict admission gates; do not auto-add video frames to known identities. |
| GPU lab drifts from production | Lab imports production services directly; no copied logic. |
| Frontend baseline is Phase 1 drift | Update only the contract-aligned hooks/pages; run `phase2-m8-ui-static`. |

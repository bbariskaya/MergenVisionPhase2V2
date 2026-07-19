# MergenVisionDemo Cross-Repo Knowledge Pack

## Executive Summary

MergenVisionDemo is a reference implementation of a GPU-accelerated face-identity system written in Python with a custom CUDA/pybind11 native extension. Its bulk-enrollment path is the strongest artifact for Phase2v2: it reads dataset folders, parses them into deterministic `person_id`/`face_identity_id`/`photo_id`/`sample_id` namespaces, and pushes images through an end-to-end GPU pipeline (JPEG decode → RetinaFace detection → alignment → ArcFace embedding → L2 normalize) while keeping almost every intermediate buffer on the device. Persistence is batched across MinIO, PostgreSQL (`insert ... on_conflict_do_update`), and Qdrant, and the work is split into idempotent shards dispatched to dedicated GPU worker processes over HTTP. The most important takeaway for Phase2v2 is that bulk enrollment is fast because it amortizes CPU→GPU transfers per batch, streams extraction and persistence with a tiny bounded queue (`maxsize=2`), uses deterministic IDs so re-runs are idempotent upserts, and keeps the heavyweight TensorRT context on a single dedicated GPU thread.

## Bulk Enrollment

- **Key files and symbols**
  - `backend/app/services/bulk_enrollment.py` — `BulkEnrollmentService` (line 85), `_extract_batch_faces` (line 125), `_read_and_extract` (line 373), `_produce` (line 427), `_consume` (line 489), `_persist_batch` (line 241), `_ensure_identities` (line 176), `_upload_photo` (line 225), `_upsert_qdrant` (line 365), `_commit_progress` (line 528), `enroll_shard` (line 568), `create_parent_process` (line 736), `finalize_parent_process` (line 746).
  - `backend/app/services/bulk_manifest.py` — `shard_by_person_id` (line 105), `build_lfw_manifest`, `build_casia_manifest`, `stream_vggface_manifest`.
  - `backend/tests/services/test_bulk_enrollment_fast.py` — fast unit/contract tests for idempotency and grouping.

- **Algorithm / flow**
  1. A parent `ProcessRecord` (`process_type="bulk_enroll"`) is created; the dataset is split into disjoint shards by `person_id % num_shards` (`shard_by_person_id`).
  2. Each shard is dispatched to a GPU worker via `POST /internal/v1/jobs` (`bulk_orchestrator.py:_dispatch_one_shard`).
  3. Inside the worker, `_load_identities` streams the shard as `EnrollmentIdentity` objects (each identity has one or more `EnrollmentPhoto` objects).
  4. `BulkEnrollmentService.enroll_shard` runs a producer/consumer pattern:
     - `_produce` fills pending pairs up to `batch_size`, calls `_read_and_extract` to read bytes concurrently and run GPU extraction, then puts the extracted chunk on an `asyncio.Queue(maxsize=2)`.
     - `_consume` takes chunks from the queue and calls `_persist_batch`.
  5. `_persist_batch` does a single idempotent persistence step per chunk:
     - `_ensure_identities`: blind bulk upsert of `FaceIdentity` and `Person` rows with `ON CONFLICT DO NOTHING` indexed by the HMAC lookup key.
     - concurrent MinIO uploads with `_upload_photo` bounded by `asyncio.Semaphore(max_persistence_concurrency)`.
     - bulk upsert of `PersonPhoto` and `FaceSample` rows with `ON CONFLICT DO UPDATE` on `photo_id`.
     - Qdrant point upsert via `_upsert_qdrant`.
  6. After each persisted chunk, `_commit_progress` flushes a progress event to `process_record.summary["progress"]` and commits the DB transaction, giving durable resume checkpoints (`last_completed_identity_key`).
  7. The parent record is finalized by `finalize_parent_process`, aggregating per-shard enrolled/failed/error counts.

- **Batch sizes and GPU usage**
  - `bulk_extract_batch_size = 256` (`backend/app/core/config.py` line 25).
  - Detection/chip staging batch is `256`; Qdrant upserts are chunked to `256` (`backend/app/infrastructure/qdrant.py` line 195).
  - Recognizer dynamic batch is capped by the TensorRT engine profile (`GpuRecognizer.max_batch`, `backend/app/ml/gpu/recognizer.py` line 47), falling back to `64`.
  - The GPU pipeline is invoked inside `asyncio.Lock` with `loop.run_in_executor(self._gpu_executor, ...)`, ensuring only one GPU batch runs at a time per process (`BulkEnrollmentService._extract_batch_faces`).

- **Producer/consumer or streaming tricks**
  - `asyncio.Queue(maxsize=2)` keeps at most two extracted chunks in memory, decoupling IO/GPU extraction from persistence without unbounded growth (`backend/app/services/bulk_enrollment.py` enroll_shard around line 600).
  - Bounded semaphore (`bulk_max_persistence_concurrency = 32`) controls concurrent MinIO uploads.
  - Decode/read failures are counted per photo and do not fail the whole batch; fatal GPU errors abort the shard with structured `fatal_code`/`fatal_stage`/`fatal_message`.

- **What Phase2v2 can copy**
  - The deterministic ID scheme (`backend/app/core/ids.py`) makes retries, resumes, and duplicate imports safe without pre-check SELECTs.
  - The producer/consumer queue + bounded persistence semaphore pattern can be reused for any stream of images or video frames.
  - The blind `ON CONFLICT DO NOTHING` identity upsert followed by `ON CONFLICT DO UPDATE` sample/photo upsert removes read-before-write races.
  - Running the GPU pipeline in a single-thread executor prevents TensorRT context corruption from concurrent Python coroutines.

## GPU / Native Runtime

- **Key files and symbols**
  - `backend/app/ml/gpu/face_pipeline.py` — `GpuFacePipeline` (line 41), `extract_batch` (line 336), `extract_bytes` (line 207), `warmup` (line 95), `_pick_largest` (line 558).
  - `backend/app/ml/gpu/decoder.py` — `JpegGpuDecoder.decode_batch` (line 95).
  - `backend/app/ml/gpu/retinaface_preprocessor.py` — `RetinaFacePreprocessor.preprocess_batch` (line 74).
  - `backend/app/ml/gpu/retinaface_postprocess.py` — `RetinaFacePostprocess.decode`, `scale_and_compact`, `pick_largest_device`.
  - `backend/app/ml/gpu/recognizer.py` — `GpuRecognizer.embed` (line 58), `max_batch` (line 47), `_embed_chunk` (line 125).
  - `backend/app/ml/gpu/alignment.py` — `GpuFaceAligner.align` (line 106), `compute_matrices`.
  - `backend/app/ml/gpu/l2_norm.py` — `l2_normalize_device` (line 17).
  - `backend/app/ml/gpu/buffer_arena.py` — `BufferArena` (line 128), `reserve`.
  - `backend/app/ml/gpu/device_tensor.py` — `DeviceTensor` (line 21).
  - `backend/app/ml/gpu/trt_device_engine.py` — `TrtDeviceEngine.infer_device` (line 102).

- **Batch inference details**
  - `extract_batch` processes a list of JPEG buffers in chunks of `max_batch` (default `256`):
    1. `JpegGpuDecoder.decode_batch` uses `nvimgcodec` to decode JPEGs directly to GPU tensors; it rejects CPU fallback (`backend/app/ml/gpu/decoder.py` line 116–119).
    2. `RetinaFacePreprocessor.preprocess_batch` builds an `ImageBatchVarShape`, resizes to `640x640`, copies into a contiguous NHWC uint8 buffer, converts to float32, applies mean/std color twist, and reformats to NCHW using `cvcuda` (`backend/app/ml/gpu/retinaface_preprocessor.py` line 74).
    3. `TrtDeviceEngine.infer_device` sets input shapes/addresses and calls `execute_async_v3` with device-resident I/O; no host copy occurs inside inference (`backend/app/ml/gpu/trt_device_engine.py` line 102).
    4. Native CUDA kernels decode RetinaFace outputs, run NMS, scale boxes back to original image resolution, and pick the largest face per image device-side (`retinaface_decode_batch`, `nms`, `scale_clip_compact`, `retinaface_pick_largest` in `backend/native/mergenvision_gpu/src/`).
    5. `GpuFaceAligner.align` computes similarity-transform matrices and warps every detected face into a `112x112` RGB float32 chip using the native `warp_align` kernel (alignment stays on GPU).
    6. `GpuRecognizer.embed` chunks face chips if they exceed the TensorRT profile max batch, runs ArcFace inference, then L2-normalizes in-place with `l2_normalize_device` (`backend/app/ml/gpu/recognizer.py` line 141; `backend/app/ml/gpu/l2_norm.py` line 17).
  - Only the final per-image bbox/landmarks/score/embedding are copied to host; intermediate detector tensors, chips, and embeddings remain device-resident.

- **Python ↔ native bridge**
  - Native code lives in `backend/native/mergenvision_gpu/` and is built with `scikit-build-core` + `pybind11` + CMake.
  - `backend/native/mergenvision_gpu/pyproject.toml` declares `scikit-build-core>=0.9` and `pybind11>=2.12` as build-system requirements.
  - `backend/native/mergenvision_gpu/CMakeLists.txt` builds `_mergenvision_gpu.so` from `.cu` kernels and `src/bindings.cpp`, links `CUDA::cudart`, uses `CMAKE_CUDA_ARCHITECTURES 75`, and installs the module under `mergenvision_gpu`.
  - `src/bindings.cpp` exposes raw pointer-based functions (`l2_normalize`, `similarity_transform`, `nms`, `scale_clip_compact`, `scale_clip_compact_xy`, `scrfd_decode_level`, `retinaface_decode_batch`, `retinaface_pick_largest`, `argsort_descending`, `warp_align`, `spin_wait_cycles`).
  - The Python layer wraps these in `DeviceTensor` objects that carry pointer, shape, dtype, device, owner lifetime, and stream; `BufferArena` pools device allocations by `(shape, dtype)` and reuses them after CUDA event fences (`backend/app/ml/gpu/buffer_arena.py` line 128).

## Persistence & Storage

- **MinIO**
  - `backend/app/infrastructure/minio.py` — `PhotoStorage` (line 18), `put_object` (line 42).
  - Object key format used for enrollment crops: `"enrollments/{person_id}/{photo_id}"` (`backend/app/services/bulk_enrollment.py` `_upload_photo` line 226).
  - `put_object` is a thin async wrapper around the MinIO SDK using `asyncio.to_thread`.

- **PostgreSQL bulk upserts**
  - `backend/app/services/bulk_enrollment.py` `_ensure_identities` (line 176) uses PostgreSQL `insert(...).on_conflict_do_nothing(index_elements=["identity_lookup_hmac"])` for `FaceIdentity` and `Person`.
  - `_persist_batch` (line 241) uses `on_conflict_do_update(index_elements=["photo_id"], set_=...)` for both `PersonPhoto` and `FaceSample`, so re-importing the same photo updates the object key, SHA, bbox, landmarks, and status atomically.
  - Resume safety comes from deterministic IDs: re-running the same shard with the same idempotency key produces the same `photo_id`/`sample_id`, turning reruns into no-op or update operations.

- **Qdrant**
  - `backend/app/infrastructure/qdrant.py` — `FaceVectorStore` (line 36), `initialize` (line 155), `upsert_batch` (line 190), `search_active` (line 205), `set_active_batch` (line 257).
  - Collection stores 512-D vectors (`embedding_dim = 512`, `backend/app/core/config.py` line 39) with cosine or Euclidean distance configured in `FaceVectorStore.__init__`.
  - `upsert_batch` validates each point and sends Qdrant requests in 256-point batches (`backend/app/infrastructure/qdrant.py` line 195).
  - Payload is minimal: `sampleId`, `photoId`, `personId`, `active`, `modelVersion` (matches Phase2v2 requirement that Qdrant not own name/metadata history).

- **Verified code snippets**
  ```python
  # BulkEnrollmentService._persist_batch — pg_insert on_conflict pattern
  await self._db.execute(
      pg_insert(PersonPhoto)
      .values(photo_rows)
      .on_conflict_do_update(
          index_elements=["photo_id"],
          set_={
              "person_id": pg_insert(PersonPhoto).excluded.person_id,
              "object_key": pg_insert(PersonPhoto).excluded.object_key,
              ...
          },
      )
  )
  ```

## Identity Model

- **Key files and symbols**
  - `backend/app/domain/models.py` — `FaceIdentity` (line 24), `Person` (line 50), `PersonPhoto` (line 94), `FaceSample` (line 138), `ProcessRecord` (line 263), `ProcessEvent`.
  - `backend/app/core/ids.py` — `derive_face_identity_id`, `derive_person_id`, `derive_photo_id`, `derive_sample_id`, `identity_hmac`, `uuid7`.

- **Design and deterministic IDs**
  - `FaceIdentity` is the canonical identity row keyed by `identity_lookup_hmac` (HMAC-SHA256 of a stable identity key such as a normalized dataset folder name). It has an auto-generated `face_identity_id` but bulk enrollment uses a deterministic UUIDv5 derived from the HMAC.
  - `Person` is a 1:1 profile linked to `FaceIdentity` via `face_identity_id`, keyed by `national_id_lookup_hmac` (also the identity HMAC). The schema has first/last name, masked national ID, and a JSONB `details` field.
  - `PersonPhoto` stores the MinIO `object_key`, `content_sha256`, status (`staged`/`active`/`failed`/`deleted`), and dimensions.
  - `FaceSample` stores the embedding evidence for one photo: detector/embedding model versions, bbox JSONB, landmarks JSONB, `quality_score`, and status.
  - `ProcessRecord` tracks every job/shard with `process_type`, `status`, `summary` JSONB, and `error_message`. Valid statuses include `pending`, `queued`, `running`, `cancel_requested`, `cancelling`, `cancelled`, `completed`, `failed`.
  - Deterministic ID derivation (from `backend/app/core/ids.py`):
    - `person_id = uuid.uuid5(PERSON_NAMESPACE, identity_hmac)`
    - `face_identity_id = uuid.uuid5(FACE_IDENTITY_NAMESPACE, identity_hmac)`
    - `photo_id = uuid.uuid5(PERSON_PHOTO_NAMESPACE, content_sha256)`
    - `sample_id = uuid.uuid5(FACE_SAMPLE_NAMESPACE, f"{photo_id}:{model_version}")`
    - Shard `process_id` is also deterministic from idempotency key (`_shard_process_id` in `bulk_orchestrator.py`).
  - This design makes the same import idempotent across reruns and across different workers, eliminating duplicate identity creation.

## Worker / API Orchestration

- **Key files and symbols**
  - `backend/app/services/bulk_orchestrator.py` — `start_vggface_job` (line 69), `start_lfw_job` (line 141), `start_casia_job` (line 209), `_build_worker_payload` (line 279), `_dispatch_one_shard` (line 308), `_persist_shard_update` (line 366), `_probe_recognition_latency` (line 381), `dispatch_shards` (line 436), `_aggregate_shards` (line 475), `request_cancellation` (line 600), `resume_vggface_job` (line 640).
  - `backend/app/workers/gpu_worker.py` — `create_worker_app` (line 381), `_lifespan` (line 311), `_load_identities` (line 236), `_run_job`, `_ensure_single_gpu_visible` (line 142).
  - `backend/app/api/routes/bulk_jobs.py` — `/vggface`, `/lfw`, `/casia`, `/{job_id}/cancel`, `/{job_id}/resume`, `/latest`.

- **How jobs are dispatched, tracked, and resumed/cancelled**
  - The main API creates a durable parent `ProcessRecord` with status `queued` and a list of shard descriptors, each carrying a deterministic `idempotency_key` (`lfw-bulk:{parent_id}:shard:{idx}`) and a deterministic `process_id` derived from that key.
  - `dispatch_shards` transitions the parent to `running`, spawns a background latency probe, and fans out `_dispatch_one_shard` calls to each worker via `httpx.AsyncClient`.
  - Each GPU worker is a separate FastAPI process listening on port `8001`. Its `POST /internal/v1/jobs` returns `202 Accepted` immediately and starts a background `_job_runner` task, so the HTTP endpoint remains responsive during long inference.
  - The worker runs exactly one job at a time (`request.app.state.current_job_id`); concurrent requests are rejected with status `failed` and event `rejected_busy`.
  - Workers poll their own `/health/ready` and the orchestrator polls each shard's `/internal/v1/jobs/{job_id}` every `_POLL_INTERVAL_SECONDS` until a terminal status.
  - Cancellation: `request_cancellation` sets the parent to `cancel_requested`, sends `POST /internal/v1/jobs/{job_id}/cancel` to every shard's worker, then sets the parent to `cancelling`. The worker sets `app.state.cancel_requested = True`; `BulkEnrollmentService.enroll_shard` checks this via `cancel_check` and raises `EnrollmentCancelled`, which marks the shard `cancelled`.
  - Resume: `resume_vggface_job` resets terminal child shards to `pending`, sets `resume_after_identity_key` from the last durable progress checkpoint, and restarts `dispatch_shards`. `stream_vggface_manifest` uses that key to skip already-processed identities.
  - Parent aggregation is read-only in `_aggregate_shards`, summing enrolled/no_face/decode_error/failed counts across shard records and computing photos-per-second rates.

## Actionable Recommendations for Phase2v2

1. **Adopt deterministic UUIDv5 IDs for all enrollment artifacts.** Base file/symbol: `backend/app/core/ids.py` (`derive_person_id`, `derive_photo_id`, `derive_sample_id`). This removes duplicate-identity races and makes retries/resumes trivial.
2. **Use a single-thread GPU executor around the TensorRT context.** Base files: `backend/app/workers/gpu_worker.py:_lifespan` (lines 311–378), `backend/app/services/bulk_enrollment.py:_extract_batch_faces` (line 125). Running inference in `ThreadPoolExecutor(max_workers=1)` with an `asyncio.Lock` prevents context corruption and simplifies synchronization.
3. **Keep decode/preprocess/infer/normalize on the GPU via `DeviceTensor` + `BufferArena`.** Base files: `backend/app/ml/gpu/device_tensor.py`, `backend/app/ml/gpu/buffer_arena.py`, `backend/app/ml/gpu/face_pipeline.py:extract_batch` (line 336). Avoid NumPy/OpenCV intermediate copies; copy only compact metadata and the final 512-D embedding to host.
4. **Stream extraction and persistence with a bounded producer/consumer queue.** Base file: `backend/app/services/bulk_enrollment.py` (`enroll_shard`, `_produce`, `_consume`, `asyncio.Queue(maxsize=2)`). Bound queue memory and decouple IO/decoding/embedding from database/storage writes.
5. **Persist per chunk and commit durable progress checkpoints.** Base file: `backend/app/services/bulk_enrollment.py:_commit_progress` (line 528). Store `last_completed_identity_key` (or frame/timestamp for video) in `ProcessRecord.summary` and commit after each successful chunk, enabling resume without reprocessing from start.
6. **Batch upsert identities, photos, samples, and Qdrant points instead of one-at-a-time inserts.** Base file: `backend/app/services/bulk_enrollment.py:_persist_batch` (line 241). Use `pg_insert(...).on_conflict_do_nothing` for first-seen identities and `on_conflict_do_update` for photo/sample evidence.
7. **Add native CUDA helpers via pybind11 + scikit-build-core, not ctypes workarounds.** Base files: `backend/native/mergenvision_gpu/pyproject.toml`, `backend/native/mergenvision_gpu/CMakeLists.txt`, `backend/native/mergenvision_gpu/src/bindings.cpp`. Keep the operators small (decode, NMS, scale, align, L2 normalize, argsort) and call them with raw device pointers from Python.
8. **For a future video pipeline, reuse the same `GpuFacePipeline` batch abstraction on sampled frames.** Base file: `backend/app/ml/gpu/face_pipeline.py:extract_batch` (line 336). Provide a list of frame byte buffers (or NVMM surfaces) and receive per-frame bboxes/landmarks/embedding; the tracker/reconciliation can then run in Python on compact metadata.
9. **Dispatch long-running recognition work to separate GPU worker processes.** Base files: `backend/app/services/bulk_orchestrator.py:dispatch_shards` (line 436), `backend/app/workers/gpu_worker.py:create_worker_app` (line 381). This keeps the online API responsive and lets you restart/cycle GPU workers without restarting the user-facing API.
10. **Enforce status-machine + idempotency in `ProcessRecord`.** Base file: `backend/app/domain/models.py:ProcessRecord` (line 263), `backend/app/workers/gpu_worker.py` (`create_job`, `cancel_job`, `get_job`). Use deterministic process IDs from idempotency keys, guard against duplicate work, and expose polling/cancel/resume endpoints.

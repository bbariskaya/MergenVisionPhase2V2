# MergenVisionDemo Cross-Repo Knowledge Pack

## Executive Summary

MergenVisionDemo is a Python/FastAPI face-recognition backend with a React frontend. Its bulk-enrollment path is built around long-lived GPU workers that own one CUDA device each, keep a warm `GpuFacePipeline` across batches, and process durable shard jobs dispatched from an API-side orchestrator. The hot path keeps JPEG decoding, detection, alignment, recognition, and L2 normalization entirely on the GPU via `DeviceTensor`s and a `BufferArena`; only the final embeddings and metadata touch the host. PostgreSQL, MinIO, and Qdrant are updated with batched upserts using deterministic, HMAC-derived UUIDs so cancel/resume is idempotent. Phase2v2 can copy the producer/consumer service, the GPU pipeline design, the native pybind11 bridge, and the persistence patterns directly.

## Bulk Enrollment

- Key files and symbols
  - `/home/user/MergenVisionDemo/backend/app/services/bulk_enrollment.py` — `BulkEnrollmentService`, `_produce`, `_consume`, `_read_and_extract`, `_extract_batch_faces`, `_persist_batch`, `_ensure_identities`, `_upload_photo`, `_upsert_qdrant`, `enroll_shard`
  - `/home/user/MergenVisionDemo/backend/app/services/bulk_manifest.py` — `EnrollmentIdentity`, `EnrollmentPhoto`, `build_lfw_manifest`, `build_casia_manifest`, `shard_by_person_id`
  - `/home/user/MergenVisionDemo/backend/app/services/bulk_orchestrator.py` — `start_vggface_job`, `start_lfw_job`, `start_casia_job`, `dispatch_shards`, `request_cancellation`, `resume_vggface_job`
  - `/home/user/MergenVisionDemo/backend/app/workers/gpu_worker.py` — GPU worker FastAPI lifespan, job endpoints, `_run_job`
  - `/home/user/MergenVisionDemo/backend/app/api/routes/bulk_jobs.py` — public `/bulk-jobs/{vggface,lfw,casia,...}` endpoints
  - `/home/user/MergenVisionDemo/backend/app/services/face_service.py` — `FaceService.bulk_enroll` (simpler HTTP-facing bulk path)
  - `/home/user/MergenVisionDemo/backend/app/core/config.py` — `bulk_extract_batch_size=256`, `bulk_max_persistence_concurrency=32`, `bulk_activation_batch_size=2048`

- Algorithm / flow
  1. `enroll_shard` creates a `ProcessRecord` for the shard.
  2. `_produce` streams identities/photos, reads bytes asynchronously, calls `_read_and_extract`, and puts extracted chunks on an `asyncio.Queue(maxsize=2)`.
  3. `_consume` takes chunks and calls `_persist_batch`, which upserts identities, uploads to MinIO, upserts Postgres rows, and upserts Qdrant points.
  4. `_commit_progress` writes durable progress after each batch so resume can start from `last_completed_identity_key`.
  5. The API orchestrator creates a parent `ProcessRecord`, splits work into shards by `person_id`, dispatches shard descriptors to GPU workers over HTTP, and polls status until terminal.

- Batch sizes and GPU usage
  - Extraction batch size defaults to `256` (`bulk_extract_batch_size`).
  - Persistence concurrency is bounded by a semaphore defaulting to `32` (`bulk_max_persistence_concurrency`).
  - The producer queue is intentionally small (`maxsize=2`) to back-pressure file reads when persistence or GPU inference is slower.
  - GPU inference is serialized per pipeline by an `asyncio.Lock`; the actual call runs in a single-worker `ThreadPoolExecutor` (`gpu_executor`) to keep CUDA work off the event loop.
  - File reads run on a separate `ThreadPoolExecutor` (`io_executor`) with `min(32, cpu_count*2)` workers.

- What Phase2v2 can copy
  - The producer/consumer split with bounded queue, separate IO and GPU executors, and per-batch progress commits.
  - The deterministic manifest/sharding scheme (`shard_by_person_id`) so the same identity always lands on the same shard.
  - The idempotent identity creation using `pg_insert(...).on_conflict_do_nothing` with deterministic UUIDs.
  - The worker/orchestrator split so GPU context lifetime is not tied to HTTP request lifetime.

## GPU / Native Runtime

- Key files and symbols
  - `/home/user/MergenVisionDemo/backend/app/ml/gpu/face_pipeline.py` — `GpuFacePipeline`, `extract_batch`, `extract_bytes`, `_scaled_to_host`
  - `/home/user/MergenVisionDemo/backend/app/ml/gpu/decoder.py` — `JpegGpuDecoder.decode_batch`
  - `/home/user/MergenVisionDemo/backend/app/ml/gpu/preprocess.py` / `retinaface_preprocessor.py` — detector preprocessors
  - `/home/user/MergenVisionDemo/backend/app/ml/gpu/retinaface_postprocess.py` / `scrfd_postprocess.py` — native CUDA postprocess + NMS + scale/compact
  - `/home/user/MergenVisionDemo/backend/app/ml/gpu/alignment.py` — `GpuFaceAligner.align`, `similarity_transform`, `warp_align`
  - `/home/user/MergenVisionDemo/backend/app/ml/gpu/recognizer.py` — `GpuRecognizer.embed`, `max_batch`
  - `/home/user/MergenVisionDemo/backend/app/ml/gpu/trt_device_engine.py` — `TrtDeviceEngine.infer_device`
  - `/home/user/MergenVisionDemo/backend/app/ml/gpu/buffer_arena.py` — `BufferArena`, `BufferLease`
  - `/home/user/MergenVisionDemo/backend/app/ml/gpu/device_tensor.py` — `DeviceTensor`
  - `/home/user/MergenVisionDemo/backend/app/ml/gpu/l2_norm.py` — `l2_normalize_device`
  - `/home/user/MergenVisionDemo/backend/native/mergenvision_gpu/src/bindings.cpp` — pybind11 `_mergenvision_gpu` module
  - `/home/user/MergenVisionDemo/backend/native/mergenvision_gpu/CMakeLists.txt` and `__init__.py` — build/package layout
  - `/home/user/MergenVisionDemo/backend/Dockerfile` — multi-stage CUDA 12.4 build

- Batch inference details
  - `extract_batch` is optimized for RetinaFace R50 dynamic batch:
    1. `JpegGpuDecoder.decode_batch` decodes JPEG buffers to GPU tensors.
    2. `RetinaFacePreprocessor.preprocess_batch` builds the detector input.
    3. `TrtDeviceEngine.infer_device` runs TensorRT detection.
    4. `RetinaFacePostprocess.decode` runs native CUDA decode/NMS.
    5. `scale_and_compact` maps detections back to original image coordinates per image.
    6. `pick_largest_device` selects the largest face per image on the GPU.
    7. Valid selections are copied to host; for each valid image, `GpuFaceAligner.align` warps a 112x112 chip on the GPU into a shared chip batch.
    8. `GpuRecognizer.embed` runs ArcFace once for the whole chip batch and native L2-normalizes on device.
    9. Only embeddings, bboxes, landmarks, and scores are copied to host at the end.
  - For non-RetinaFace packs (e.g., `antelopev2`), `extract_batch` falls back to per-image `extract_bytes` with `_pick_largest`.
  - `GpuRecognizer.max_batch()` reads the TensorRT engine's optimization profile max shape; it defaults to `64` if unavailable. Larger recognizer batches are chunked internally.
  - `BufferArena` pools GPU allocations keyed by shape/dtype and fences reuse with CUDA events (`BufferLease.release` records an event; `acquire` only reuses completed allocations).

- Python ↔ native bridge
  - Native code is built as a pybind11 module named `_mergenvision_gpu` and packaged under `mergenvision_gpu`.
  - Python imports operators directly: `from mergenvision_gpu import l2_normalize, similarity_transform, warp_align, ...`.
  - Every function accepts integer device pointers (`uintptr_t`) and a stream handle; Python passes `tensor.ptr` and the pipeline's stream.
  - Error status is written into a `DeviceTensor` (`[1] int32`) by the kernel and read back after `cudaStreamSynchronize`.
  - The Dockerfile compiles the extension in a `nvidia/cuda:12.4.1-devel-ubuntu22.04` builder stage and copies the `.so` into the runtime image.

## Persistence & Storage

- MinIO, Postgres, Qdrant patterns
  - **MinIO**: `/home/user/MergenVisionDemo/backend/app/infrastructure/minio.py` — `PhotoStorage.put_object` stores raw bytes with key `enrollments/{person_id}/{photo_id}` using `asyncio.to_thread` around the MinIO SDK. Bucket creation is lazy in `initialize`.
  - **PostgreSQL bulk upserts**: `/home/user/MergenVisionDemo/backend/app/services/bulk_enrollment.py` `_persist_batch`:
    - `FaceIdentity` and `Person` are inserted with `pg_insert(...).on_conflict_do_nothing(index_elements=[...])`.
    - `PersonPhoto` and `FaceSample` are inserted with `pg_insert(...).on_conflict_do_update(index_elements=["photo_id"], set_=...)` so re-running the same photo updates rather than duplicates.
    - All writes happen in one batch per chunk; `_commit_progress` calls `await db.commit()` after each activation batch.
  - **Qdrant**: `/home/user/MergenVisionDemo/backend/app/infrastructure/qdrant.py` — `FaceVectorStore.upsert_batch`:
    - Validates payload keys (`sampleId`, `photoId`, `personId`, `active`, `modelVersion`) and vector shape/finiteness.
    - Sends points in 256-point sub-batches (`batch_size = 256`) with `wait=False` by default for bulk.
    - `search_active` filters by `active=True` and `modelVersion` using payload indexes.
    - `set_active_batch` performs soft deletes by toggling the `active` payload field.

- Bulk upsert examples
  - Identity upsert:
    ```python
    await self._db.execute(
        pg_insert(FaceIdentity)
        .values(face_rows)
        .on_conflict_do_nothing(index_elements=["identity_lookup_hmac"])
    )
    ```
  - Photo upsert:
    ```python
    await self._db.execute(
        pg_insert(PersonPhoto)
        .values(photo_rows)
        .on_conflict_do_update(
            index_elements=["photo_id"],
            set_={
                "person_id": pg_insert(PersonPhoto).excluded.person_id,
                "object_key": pg_insert(PersonPhoto).excluded.object_key,
                "content_sha256": pg_insert(PersonPhoto).excluded.content_sha256,
                "status": "active",
                "updated_at": now,
            },
        )
    )
    ```
  - Qdrant point payload mirrors the Postgres sample ID: `sampleId == point.id`.

## Identity Model

- Person/FaceIdentity/FaceSample design
  - `/home/user/MergenVisionDemo/backend/app/domain/models.py` defines the schema:
    - `FaceIdentity` — canonical identity row keyed by `identity_lookup_hmac` (unique), links to many `Person`s via `face_identity_id` FK.
    - `Person` — per-dataset/person record with `national_id_lookup_hmac` (unique), `face_identity_id` FK, JSONB `details`, soft-delete via `deleted_at`.
    - `PersonPhoto` — one row per stored photo; `object_key` unique in MinIO; `status` in `('staged','active','failed','deleted')`; unique `(person_id, content_sha256)`.
    - `FaceSample` — one row per extracted face; `photo_id` unique FK; stores `bbox` and `landmarks` as JSONB, `detector_model`, `embedding_model`, `quality_score`, `status`.
  - Deterministic IDs live in `/home/user/MergenVisionDemo/backend/app/core/ids.py`:
    - `identity_hmac(identity_key, master_key)` → HMAC-SHA256.
    - `derive_person_id(hmac)` and `derive_face_identity_id(hmac)` are UUIDv5 over fixed namespaces.
    - `derive_photo_id(content_sha256)` is UUIDv5 over the photo hash.
    - `derive_sample_id(photo_id, model_version)` is UUIDv5 over `photo_id:model_version`.
  - This design makes re-imports idempotent: the same folder/photo/model always produces the same UUIDs, and `ON CONFLICT DO NOTHING/UPDATE` handles collisions without pre-SELECTs.

## Actionable Recommendations for Phase2v2

1. Reuse the `BulkEnrollmentService` producer/consumer architecture: bounded `asyncio.Queue`, separate GPU and IO executors, and per-batch progress commits to survive resume/cancel.
2. Port the deterministic ID scheme from `app/core/ids.py` so bulk imports, duplicates, and resume checkpoints are idempotent without existence SELECTs.
3. Copy `GpuFacePipeline.extract_batch` end-to-end for the still-image hot path: batched nvImageCodec decode → TensorRT detector → native CUDA decode/NMS → GPU alignment → batched ArcFace → native L2 normalize, keeping all intermediate data as `DeviceTensor`s.
4. Adopt `BufferArena` + `BufferLease` with CUDA event fences instead of allocating/freeing GPU memory per batch; this is the core mechanism that keeps the pipeline fast and allocation-stable.
5. Use `TrtDeviceEngine.infer_device` style DeviceTensor binding so no H2D/D2H copies occur inside the inference hot path.
6. Mirror the persistence triple: PostgreSQL `pg_insert` upserts for identity/photo/sample, MinIO upload under a bounded semaphore, and Qdrant 256-point upsert batches with strict payload validation.
7. Build a single pybind11 native package (e.g., `mergenvision_gpu`) exposing only pointer/stream functions, compiled in a CUDA-devel Docker builder stage and installed into the runtime image.
8. Use the worker/control-plane split: API orchestrator owns durable `ProcessRecord`s, dispatches compact shard descriptors to GPU workers over HTTP, and polls status; this keeps GPU context lifetime separate from HTTP request lifetime.
9. For a future video pipeline, MergenVisionDemo has no video decoder; add a GPU video decode stage (NVIDIA Video Codec SDK / PyNvVideoCodec) and feed extracted frame batches into the same `extract_batch` path.
10. Reuse the Qdrant payload schema (`active` bool + `modelVersion` keyword) and `set_active_batch` soft-delete pattern so sample lifecycle changes do not require vector re-upserts.

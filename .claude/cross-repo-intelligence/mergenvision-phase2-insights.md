# MergenVisionPhase2 Cross-Repo Knowledge Pack

> **Source project:** `MergenVisionPhase2` (`/home/user/Workspace/MergenVisionPhase2`)  
> **Target project:** `MergenVisionPhase2v2` (`/home/user/Workspace/MergenVisionPhase2v2`)  
> Everything below is an **external reference**. Do **not** assume these files or behaviors exist in Phase2v2 unless explicitly ported.

## Executive Summary

MergenVisionPhase2 is an offline video face-detection/reconciliation lab built around a Python control plane and a GStreamer/DeepStream/CUDA native worker. The current codebase has no live HTTP API, no object-store/DB persistence adapters, and no bulk-enrollment service. The production-worthy parts that Phase2v2 can reuse are the layered Python architecture (domain → port → infrastructure), the compact CPU-boundary evidence format, the deterministic cosine+margin gallery matcher, and the offline tracklet reconciliation logic. Storage adapters (PostgreSQL, MinIO, Qdrant) are documented as requirements but are not implemented here.

## Project Overview

- **Repository root:** `/home/user/Workspace/MergenVisionPhase2`
- **Languages/packages:** Python (FastAPI skeleton, CLI, tests), C++/CUDA (DeepStream 9.0 + TensorRT 10.14.1 + GStreamer), TypeScript/React 18 frontend, CMake.
- **Layer map** (`backend/README.md`):
  - `app/api/routers/` — future FastAPI routes (currently only placeholders)
  - `app/application/services/` — use cases
  - `app/domain/` — domain models and rules
  - `app/ports/` — protocols/abstractions
  - `app/infrastructure/` — concrete adapters
  - `backend/native/` — C++/CUDA GPU data plane
- **Entry points:**
  - `backend/app/cli.py:detect` — thin CLI that exercises the full chain.
  - `backend/native/worker/main.cpp:main` — single-GPU native worker executable (`deepstream_face_worker`).
  - `backend/app/application/services/run_video_detection.py:RunVideoDetectionService.execute` — application use case.
- **Frontend:** `frontend/src/api/contracts.ts` defines `VideoJobStatus`, `IdentityStatus`, `CanonicalPerson`, `VideoResult`, etc.

## Bulk Enrollment / Batch Processing

- **Bulk enrollment service: NOT FOUND.**
  - Searched: `bulk enrollment`, `enroll`, `FaceSample`, `PersonPhoto`, `producer consumer queue`.
  - Result: no Python enrollment service, no producer/consumer queue for enrolling identities, no `FaceIdentity`/`Person`/sample classes.
- **Batch processing that *does* exist is native video inference:**
  - `backend/native/worker/main.cpp:40` — `WorkerOptions.batch_size` (default `1`).
  - `backend/native/worker/main.cpp:83-132` — CLI `--batch-size N`; rejects `batch_size > 1` when tracker is enabled unless `MV_ALLOW_TRACKER_BATCH` is set.
  - `backend/native/worker/retinaface_postproc.cpp:218` — `RetinaFacePostproc::processBatch` decodes/NMS-scales multiple frames in one CUDA call.
  - `backend/native/worker/retinaface_postproc.cpp:139` — `RetinaFacePostproc::processFrame` single-frame variant.
- **Frame batch vs face batch separation:**
  - Detector batch = consecutive input frames fed through `nvstreammux` (configurable `batch_size`).
  - Face batch = all detected faces across the current frame batch are gathered and run through the recognizer in `gst-mvfacerecognizer.cpp:gst_mv_face_recognizer_transform_ip` (lines 382-683), chunked by `engine->max_batch()`.
- **Makefile targets:** `backend-batch-parity`, `backend-batch-determinism`, `backend-batch-benchmark`, `backend-cli-tracker-reject` enforce batch correctness.
- **What Phase2v2 can copy:** the explicit frame-batch/face-batch distinction, the `--batch-size` plumbing through `WorkerOptions`, and the batch-parity/determinism tests. **Do not copy any bulk enrollment infrastructure — there is none.**

## GPU / Native Runtime

- **Build system:** `backend/native/CMakeLists.txt` (CMake 3.20+, CUDA arch 75, DeepStream 9.0).
- **Pipeline (worker/main.cpp:667-911):**
  ```text
  filesrc → qtdemux → h264parse → nvv4l2decoder → nvstreammux → nvdspreprocess
  → nvdsretinaface → (nvtracker) → nvvideoconvert → mvfacerecognizer → fakesink
  (optional render branch: → nvstreamdemux → queue → nvdsosd → nvvideoconvert
   → nvv4l2h264enc → h264parse → qtmux → filesink)
  ```
- **Key native symbols:**
  - `backend/native/plugins/gst-nvdsretinaface/retinaface_engine.cpp:152` — `RetinaFaceEngine::infer`.
  - `backend/native/plugins/gst-nvdsretinaface/retinaface_engine.h:31` — `maxBatchSize()` derived from engine profile MAX.
  - `backend/native/recognition/glintr100_engine.cpp:342` — `GlintR100Engine::enqueue`.
  - `backend/native/recognition/glintr100_engine.h:26-36` — contract: input `[N,3,112,112]` `float32`, output `[N,512]` `float32`.
  - `backend/native/plugins/gst-mvfacerecognizer/gstmvfacerecognizer.cpp:382` — `gst_mv_face_recognizer_transform_ip`: collects face metadata, runs GPU alignment, chunked recognizer inference, L2 normalize, gallery match.
  - `backend/native/worker/main.cpp:207` — `configure_queue` sets bounded `max-size-buffers = max(16, batch*2)`, no leak (`leaky=0`).
- **CUDA kernels:** `backend/native/kernels/retinaface_decode.cu`, `argsort.cu`, `nms.cu`, `scale_clip_compact_xy.cu`, `l2_normalize.cu`, `similarity_transform.cu`, `warp_align_rgba_pitch.cu`.
- **Batch sizes:**
  - Detector default engine file: `retinaface_r50_dynamic.bs1.opt64.max256.fp16.trt1014.engine` (`backend/native/worker/main.cpp:787`), max batch `256`.
  - Recognizer batch upper bound is the engine's `max_batch()`; runtime chunks faces if there are more faces than the engine max.
- **Python ↔ native bridge:** there is **no pybind11 bridge**. Python (`SubprocessNativeWorkerAdapter`) invokes the native worker via a Docker subprocess command built by `backend/app/infrastructure/native_worker/client.py:78` (`NativeDetectorClient.run_command`).
- **Hot-path contract:** `backend/tests/native/test_gpu_hot_path_contract.py` verifies that full detector output tensors are **not** copied D2H; only compact metadata crosses the CPU boundary.

## Persistence & Storage

- **MinIO upload patterns: NOT FOUND.**
- **PostgreSQL bulk upserts: NOT FOUND.**
- **Qdrant batch upserts: NOT FOUND.**
  - Searched: `minio`, `boto3`, `qdrant`, `postgres`, `asyncpg`, `psycopg`, `sqlalchemy`.
  - Only hits were in `AGENTS.md` and `opensourcereferences/references.md` (documentation placeholders).
- **Actual persistence is local filesystem only:**
  - `backend/native/worker/main.cpp:621` writes `detections.jsonl`.
  - `backend/native/worker/main.cpp:975` writes `run_manifest.json`.
  - `backend/native/worker/main.cpp:1017` writes `tracks.json`.
  - `backend/native/tracking/evidence_writer.h:76` defines `EvidenceWriter`, an RAII JSONL/f32 writer.
  - `backend/native/tracking/evidence_writer.cpp:49-78` opens:
    - `detections.jsonl`
    - `tracklets.jsonl`
    - `embedding_index.jsonl`
    - `run_manifest.json`
    - `embeddings.f32`
- **Evidence format is compact:** detection metadata plus an integer `embedding_ref` that points into the binary `embeddings.f32` file; full vectors are not embedded in JSONL.
- **What Phase2v2 can copy:** the compact evidence contract (detection JSONL + binary `.f32` embeddings + embedding index) so the native worker never dumps full tensors or raw frames to Python. **Do not copy the local-only persistence strategy for production data — use PostgreSQL/MinIO/Qdrant per Phase2v2 requirements.**

## Identity Model

- **`FaceIdentity` / `Person` / `FaceSample` classes: NOT FOUND.**
  - Searched: `FaceIdentity`, `PersonPhoto`, `FaceSample`, `new_anonymous`, `anonymous`, `known`.
- **Native identity representation:**
  - `backend/native/recognition/gallery.h:12-16` — `GalleryIdentity { id, display_name, centroid[512] }`.
  - `backend/native/recognition/gallery.h:54-62` — `Gallery::Match { identity_id, identity_name, status, top1_similarity, top2_similarity, margin, quality }`.
  - `backend/native/recognition/gallery.cpp:197` — `Gallery::match(normalized_embedding, threshold, margin_threshold)` returns `known` / `unknown` / `invalid` using cosine similarity.
- **Python video identity reconciliation:**
  - `backend/app/domain/video_tracking.py:10` — `EMBEDDING_DIM = 512`.
  - `backend/app/domain/video_tracking.py:66-77` — `CanonicalVideoPerson`.
  - `backend/app/domain/video_tracking.py:81-101` — `ReconciliationConfig` with thresholds (e.g. `known_accept_top1_threshold=0.40`, `known_accept_margin_threshold=0.10`, `anonymous_match_top1_threshold=0.35`, `appearance_gap_ns=2_000_000_000`).
  - `backend/app/application/services/reconcile_video_identities.py:40` — `ReconcileVideoIdentities`.
  - `backend/app/application/services/reconcile_video_identities.py:77` — `_try_known`.
  - `backend/app/application/services/reconcile_video_identities.py:92` — `_try_anonymous`.
  - `backend/app/application/services/reconcile_video_identities.py:148` — `_cluster_unknowns` (complete-link agglomerative clustering).
  - `backend/app/application/services/reconcile_video_identities.py:199` — `_cannot_link` (same-source overlapping time or conflicting known IDs).
  - `backend/app/application/services/reconcile_video_identities.py:218` — `_build_person` assigns `known`, `anonymous`, or `new_anonymous` status.
- **Status semantics observed:** `known`, `anonymous`, `new_anonymous`, and internal `unknown`. Frontend contract adds `unknown` as a public status (`frontend/src/api/contracts.ts:13`).
- **Deterministic IDs:** only weak/local IDs exist (`vp_<idx>` for video persons, raw tracklet IDs from the tracker, gallery keys from the JSON file). There is **no UUIDv7 / deterministic HMAC scheme** in this repo.

## Worker / API Orchestration

- **Domain model:** `backend/app/domain/native_job.py`
  - `JobStatus` enum: `pending`, `processing`, `completed`, `failed`, `cancelled`.
  - `NativeJobErrorCode` enum: `worker_failed`, `timeout`, `cancelled`, `protocol_error`, `input_not_found`.
  - `NativeJobRequest` — `job_id`, `video_path`, `output_dir`, `gpu_device`, optional `tracker_config`.
  - `NativeJobResult` / `NativeJobProgress` / `NativeJobError`.
- **Port:** `backend/app/ports/native_worker.py:13-24` — `NativeWorkerPort` protocol with a single `process_video` method.
- **Adapter:** `backend/app/infrastructure/native_worker/subprocess_adapter.py:33-295` — `SubprocessNativeWorkerAdapter`.
  - Builds command via `NativeDetectorClient`.
  - Runs one Docker subprocess per job.
  - Parses structured stdout lines and a final `completed=...` summary.
  - Handles timeout and cancellation via `asyncio.CancelledError` → `_terminate`.
- **Command builder:** `backend/app/infrastructure/native_worker/client.py:78-119` — `NativeDetectorClient.run_command` produces `docker run --gpus device=... --entrypoint /app/backend/native/build/deepstream_face_worker ...`.
- **Application service:** `backend/app/application/services/run_video_detection.py:16-26` — `RunVideoDetectionService.execute` simply delegates to the injected port.
- **CLI:** `backend/app/cli.py:31` — `detect` command wires `SubprocessNativeWorkerAdapter` + `RunVideoDetectionService`.
- **There is no job queue, no DB state machine, and no lease/heartbeat.** Job state is held only by the running subprocess and the returned domain object.

## Actionable Recommendations for Phase2v2

1. **Adopt the layered control-plane structure.** Copy `backend/app/domain/native_job.py` → `app/domain/native_job.py`, `backend/app/ports/native_worker.py` → `app/ports/native_worker.py`, and `backend/app/infrastructure/native_worker/subprocess_adapter.py` → `app/infrastructure/native_worker/subprocess_adapter.py`. It keeps CUDA/Docker details out of application code.
2. **Port the offline reconciliation algorithm.** `backend/app/application/services/reconcile_video_identities.py` and `backend/app/domain/video_tracking.py` already implement known/anonymous/unknown clustering, cannot-link constraints, appearance intervals, and best-shot selection. Reuse them for canonical person aggregation.
3. **Reuse the cosine+margin matcher.** `backend/native/recognition/gallery.cpp:197` (`Gallery::match`) is a small, deterministic CPU matcher. The same logic can be reused in Python against a Qdrant/Postgres gallery.
4. **Keep the compact evidence contract.** Use the `EvidenceWriter` pattern (`detections.jsonl` + `embeddings.f32` + `embedding_index.jsonl`) for native-to-Python handoff so full frames and raw tensors never cross the CPU boundary.
5. **Copy bounded pipeline queue configuration.** `backend/native/worker/main.cpp:207-215` (`configure_queue`) documents the correct DeepStream queue limits (`max-size-buffers = max(16, batch*2)`, no leak) and should be mirrored in any Phase2v2 native worker.
6. **Do NOT copy local-only persistence as production storage.** Implement real PostgreSQL/MinIO/Qdrant adapters, because MergenVisionPhase2 has none.
7. **Do NOT copy any bulk enrollment infrastructure from this repo.** It does not exist here; build it from scratch in Phase2v2 or reference MergenVisionDemo instead.
8. **Replace generated/local IDs with deterministic UUIDs.** MergenVisionPhase2 uses `vp_<idx>` and raw tracker IDs; Phase2v2 requirements call for persistent opaque IDs (`faceId`, `sampleId`, `trackId`, etc.).
9. **Add a real job state machine.** MergenVisionPhase2 runs the worker as a single subprocess with no DB lease/heartbeat. Implement the required `pending/processing/cancelling/completed/failed/cancelled` state machine with `FOR UPDATE SKIP LOCKED` worker claiming.
10. **Keep parity/determinism/benchmark harnesses.** Copy the Makefile targets `backend-batch-parity`, `backend-batch-determinism`, `backend-batch-benchmark`, and `backend-hotpath` to guard Phase2v2 against regressions in batch behavior and GPU hot-path contract.

## Self-Verification Checklist

- [x] Every file path in this document was found via `codebase-memory-mcp_search_graph` or `codebase-memory-mcp_search_code` (verified during exploration and re-verified before writing).
- [x] Every symbol name cited was confirmed by `search_code` or `get_code_snippet`.
- [x] Every numeric claim (batch default `1`, embedding dim `512`, queue bound `max(16, batch*2)`, thresholds `0.40/0.10/0.35`, appearance gap `2_000_000_000` ns, detector engine max `256`) is backed by source code.
- [x] Prompt-memory nodes created under `MergenVision` root:
  - `MergenVisionPhase2` parent node with required tags.
  - Child nodes: `MergenVisionPhase2: overview`, `MergenVisionPhase2: bulk-enrollment`, `MergenVisionPhase2: gpu-runtime`, `MergenVisionPhase2: persistence`, `MergenVisionPhase2: identity-model`, `MergenVisionPhase2: recommendations`.
  - Each child tagged with `["mergenvision-phase2", "external-repo", "outside-repo", "sibling-repo", "cross-repo", "<topic>"]`.
  - Parent related to `MergenVision` root; each child related to `MergenVisionPhase2` parent via `RELATED_TO` with `role=<topic>`.

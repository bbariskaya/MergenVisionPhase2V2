# GPU Video Worker Design

Date: 2026-07-17
Decision: Approach A — one native observation worker binary per GPU + Python reader/orchestrator.

## Goal

Make queued video jobs in the PostgreSQL `video_job` table actually execute on the GPU. The existing FastAPI backend only uploads videos and queues jobs; no process claims them. This design adds that worker process and supports 3 concurrent jobs on 3 GPUs.

## Architecture

```text
                    docker-compose.gpu.yml
mergenvision-backend-gpu       worker-gpu-0      worker-gpu-1      worker-gpu-2
        |                            |                 |                 |
   POST /api/v1/videos/recognize      |                 |                 |
        |                            |                 |                 |
        v                            v                 v                 v
  PostgreSQL                 pg-claim-loop       pg-claim-loop      pg-claim-loop
  video_job table            (one GPU each)      (one GPU each)     (one GPU each)
        ^                            |                 |                 |
        |                            |                 |                 |
   update job state                 v                 v                 v
                              mv_video_worker     mv_video_worker    mv_video_worker
                              (DeepStream/        (DeepStream/       (DeepStream/
                               TensorRT)           TensorRT)          TensorRT)
                                   |                 |                 |
                                   v                 v                 v
                              observations.*.pb.zst  ...               ...
                                   |                 |                 |
                                   v                 v                 v
                         VideoProcessingService.process(frames)
                         tracking -> reconciliation -> identity -> overlay
```

The backend container keeps doing what it does today: control plane, API, orchestration, persistence. New worker containers own the GPU hot path.

## Components

### 1. Native executable `mv_video_worker`

Location: `backend/native/video_worker/build/mv_video_worker`

CLI:

```text
mv_video_worker \
  --input <local_video_path> \
  --gpu-id <int> \
  --output <output_directory> \
  --detector-batch-size <int> \
  --recognizer-batch-size <int> \
  --model-profile <json_path> \
  --detector-engine <engine_path> \
  --recognizer-engine <engine_path>
```

Behavior:
- Runs the same GStreamer/DeepStream/TensorRT pipeline proven by `real_batching_smoke`.
- Writes one or more `observations.{sequence}.pb.zst` files.
- Each record is the existing `VideoObservationFrame` message.
- Emits one record per sampled frame, even when `detections` is empty.
- Writes an `ObservationChunkFooter` at the end of the final chunk.
- Exits 0 on success, non-zero on any fatal error.
- Does **not** run tracking, reconciliation, identity resolution, or overlay rendering. Its only job is compact observation emission.

### 2. Python worker `video_worker_main`

Location: `backend/app/worker/video_worker_main.py`

Responsibilities:
1. Connect to PostgreSQL using the same async SQLAlchemy stack as the API.
2. Loop forever with `asyncio.sleep(1)` between claim attempts.
3. Call `video_job_queue.claim_next(worker_id, lease_token, lease_expires_at)`.
4. On claim:
   - Fetch the source video from MinIO to a temporary local path.
   - Build the native CLI.
   - Spawn the native worker with `asyncio.create_subprocess_exec`.
   - Heartbeat the lease every few seconds while the native process runs.
   - If the DB record shows `cancellation_requested`, terminate the subprocess and mark the job `cancelled`.
   - On native success, read the `.pb.zst` artifact, decompress with `zstandard`, decode protobuf, and map to `VideoObservationFrame` DTOs.
   - Call `VideoProcessingService.process(job_id, frames)` to run tracking, reconciliation, identity, persistence, and overlay generation.
   - Mark the job `completed`.
5. On native non-zero exit or exception, call `release_for_retry` until `max_attempts`, then mark `failed`.

### 3. Worker container image

Use a dedicated image `mergenvision-worker:gpu` built from `backend/Dockerfile.worker.gpu`:

- Base: `mergenvision/deepstream-dev:9.0`
- Installs Python 3.12 dependencies from `backend/requirements.lock` and `pyproject.toml` (`pip install -e .`).
- Copies the repository so `/workspace/backend` is present.
- Builds `mv_video_worker` during image build (`cmake ... && cmake --build`).
- Entrypoint: `python -m app.worker.video_worker_main`

This keeps the DeepStream/TensorRT runtime and the Python control-plane packages in one image, while the backend API container stays on the TensorRT-only image.

### 4. `docker-compose.gpu.yml` additions

Add three worker services using `mergenvision-worker:gpu`:

```yaml
worker-gpu-0:
  image: mergenvision-worker:gpu
  container_name: mergenvision-worker-gpu-0
  working_dir: /workspace/backend
  command: python -m app.worker.video_worker_main
  environment:
    DATABASE_URL: postgresql+asyncpg://mergenvision:mergenvision@postgres:5432/mergenvision
    MINIO_ENDPOINT: minio:9000
    MINIO_ACCESS_KEY: minioadmin
    MINIO_SECRET_KEY: minioadmin
    MINIO_SECURE: "false"
    MINIO_BUCKET_NAME: mergenvision-face-samples
    QDRANT_URL: http://qdrant:6333
    QDRANT_COLLECTION_NAME: face_samples_retinaface_r50_glintr100_v1
    MODEL_VERSION: retinaface_r50_glintr100_v1
    MODEL_PROFILE_PATH: /workspace/backend/config/model_profiles/retinaface_r50_glintr100_v1_deepstream9.json
    DETECTOR_ENGINE_PATH: /workspace/backend/artifacts/engines/retinaface_r50_dynamic.bs1.opt8.max64.fp16.trt1016.engine
    RECOGNIZER_ENGINE_PATH: /workspace/backend/artifacts/engines/glintr100.bs1.opt8.max64.fp16.trt1016.engine
    GPU_DEVICE_ID: "0"
    WORKER_ID: "worker-gpu-0"
    LOG_LEVEL: INFO
  runtime: nvidia
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ["0"]
            capabilities: [gpu]
  depends_on:
    postgres:
      condition: service_healthy
    minio:
      condition: service_healthy
    qdrant:
      condition: service_healthy

worker-gpu-1: ... GPU 1
worker-gpu-2: ... GPU 2
```

Each worker pins itself to exactly one GPU. `device_ids` ensures GPU isolation.

### 5. Protobuf observation reader

Location: `backend/app/infrastructure/serialization/video_observation_reader.py`

Responsibilities:
- Read `.pb.zst` chunk files.
- Decode `VideoObservationFrame` and `ObservationChunkFooter` messages.
- Map protobuf fields to `app.application.ports.video_observations.FaceObservation` and `VideoObservationFrame`.
- Validate that the chunk footer matches the read frame count and job id.
- Raise a clear error on schema/version mismatch.

The backend's existing `video_observations.py` DTOs are the canonical contract; the reader is a thin adapter.

## Data Flow for a Single Job

1. User uploads video via `POST /api/v1/videos/recognize`.
2. Backend inserts `video_asset` and `video_job` with `state = pending`, `stage = queued`.
3. `worker-gpu-X` claims the job, setting `lease_owner`, `lease_token`, `lease_expires_at`, `state = processing`.
4. Worker downloads source video to a local temp path.
5. Worker runs `mv_video_worker`.
6. Native code decodes on NVDEC, detects with RetinaFace R50 TensorRT, aligns/recognizes with GlintR100 TensorRT, and writes protobuf artifacts.
7. Worker reads artifacts and calls `VideoProcessingService.process(job_id, frames)`.
8. Python service builds tracklets, reconciles, resolves identities, persists people, and writes the overlay timeline.
9. Worker updates `video_job` to `completed`, clears the lease.
10. User polls `GET /api/v1/videos/jobs/{jobId}` and sees `completed`.

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Native OOM/segfault | Non-zero exit; worker increments `attempt_no`, reschedules with backoff, or marks `failed` if `max_attempts` exceeded. |
| Lease expires without heartbeat | `recover_expired_leases` resets the job; another worker claims it. |
| Cancellation requested | Worker terminates subprocess, cleans temp files, sets `state = cancelled`. |
| Corrupt artifact / protobuf mismatch | Worker marks job `failed` with error code `INVALID_OBSERVATION_ARTIFACT`. |
| No observations emitted | Valid completion with zero detections; `person_count = 0`. |

## Testing

1. **Unit**: protobuf reader mapping tests with synthetic `.pb.zst` bytes.
2. **Integration**: queue a job in a test database, run `video_worker_main` against a fake native executable that writes a known artifact, assert the job finishes and result API returns expected people.
3. **Native smoke**: extend `make phase2-m6-native-full-observation` to also run `mv_video_worker --input test_videos/Friends.mp4` and assert the artifact exists and matches frame count.
4. **Compose smoke**: `make phase2-worker-gpu-smoke` brings up `worker-gpu-0` only, uploads `Friends.mp4`, and waits for `completed`.

## Out of Scope

- Moving tracking/reconciliation/identity logic into C++.
- Autoscaling workers beyond the 3 pinned containers.
- GPU sharing / fractional GPU scheduling.
- Annotated MP4 rendering.

## References

- `backend/contracts/video_observation_v1.proto`
- `backend/app/application/ports/video_observations.py`
- `backend/app/application/services/video_processing_service.py`
- `backend/app/infrastructure/persistence/sqlalchemy/video_job_queue.py`
- `backend/native/video_worker/CMakeLists.txt`
- `backend/native/video_worker/tests/real_batching_smoke.cpp`
- `docs/implementation/CURRENT_SPRINT.md`

# GPU Video Worker Implementation Plan

> **Scope update (2026-07-17):** Task 4 originally proposed an *FFmpeg track crop
> provider*.  That task is **cancelled**.  No FFmpeg, ffprobe, OpenCV
> `VideoCapture`, CPU full-frame decode, or second-pass video seek may exist in
> the production worker/runtime path.  Representative face crops must be exported
> from the aligned 112×112 GPU crop already produced inside the native pipeline
> when GlintR100 runs.

**Goal:** Make queued video jobs execute end-to-end on real GPUs with a native
DeepStream/TensorRT observation worker that emits observations + per-raw-track
templates + representative aligned crops, plus a Python claim/orchestration
worker that reads those artifacts and runs tracking/reconciliation/identity/overlay.

**Architecture:** One `mergenvision-worker:gpu` container per GPU claims jobs from
PostgreSQL, downloads the source video from MinIO, runs the native
`mv_video_worker` binary, reads the resulting artifact bundle, runs Python domain
services, and commits the job state.

**Tech Stack:** Python 3.12, FastAPI/SQLAlchemy/asyncpg, MinIO, Qdrant, Protobuf,
zstd, C++17/CUDA, GStreamer/DeepStream 9.0, TensorRT, libwebp (bounded crop
encode only), Docker Compose.

## Global Constraints

- One worker container owns exactly one GPU (`device_ids`).
- Native pipeline decodes the video exactly once via NVDEC; no second decode.
- Representative crop must be the 112×112 aligned GPU crop produced before
  GlintR100 inference.
- Full-frame GPU→CPU transfer is forbidden. Only compact 112×112 crop D2H and
  protobuf metadata may cross to CPU.
- Native worker outputs: UI-safe observation artifact, internal raw-track template
  artifact, representative crop bundle, and manifest.
- Persistent IDs are UUIDv7.
- No annotated MP4 primary output.
- Add/update tests for every deliverable; run `make phase2-step0-static` before
  declaring a task done.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `backend/contracts/video_observation_v1.proto` | UI-safe frame/detections + chunk footer. |
| `backend/contracts/video_track_template_v1.proto` | Internal raw-track template artifact. |
| `backend/native/video_worker/src/mv_video_worker.cpp` | Native executable: decode → detect → align → embed → tracker → observations/templates/crops/manifest. |
| `backend/native/video_worker/src/representative_crop_selector.cpp/hpp` | Bounded per-track candidate selection (NEW). |
| `backend/native/video_worker/CMakeLists.txt` | Builds `mv_video_worker`; links protobuf + libwebp + pipeline. |
| `backend/native/video_worker/generated/*.pb.{cc,h}` | Generated C++ protobuf sources. |
| `backend/Dockerfile.worker.gpu` | Worker image based on `mergenvision/deepstream-dev:9.0` with Python deps + protobuf + libwebp. |
| `backend/app/infrastructure/serialization/` | Python readers for observation + template + manifest artifacts. |
| `backend/app/infrastructure/runtime/native_representative_crop_provider.py` | Reads native output bundle; maps `raw_track_key → crop bytes`. |
| `backend/app/worker/video_worker_main.py` | Async claim loop, native subprocess orchestration, artifact reading, service invocation. |
| `backend/app/worker/bootstrap.py` | Wires `VideoProcessingService` and dependencies. |
| `docker-compose.gpu.yml` | Adds `worker-gpu-0/1/2` services. |
| `Makefile` | Adds worker image, native build, GPU smoke, and E2E targets. |

---

## Task List (corrected order)

### [✓] Task 1: Add Python `zstandard` dependency
- `backend/pyproject.toml`: add `zstandard`.

### [✓] Task 2: Generate Python protobuf module for observations
- `backend/app/infrastructure/serialization/video_observation_v1_pb2.py`, `__init__.py`.

### [✓] Task 3: Implement protobuf observation reader
- `backend/app/infrastructure/serialization/video_observation_reader.py`.
- Unit tests cover frames, detections, and footer counts.

### [•] Task 4: Define/complete native template + representative-crop artifact contract
- Create: `backend/contracts/video_track_template_v1.proto`.
- Generate Python pb2 and C++ pb.cc/pb.h.
- Add static forbidden-path test for `ffmpeg`, `ffprobe`, `subprocess`,
  `cv2.VideoCapture`, `PIL.Image` full-frame decode in production worker source.

### [ ] Task 5: Implement `mv_video_worker` native binary output
- Modify/create: `backend/native/video_worker/src/mv_video_worker.cpp` from the
  proven `real_batching_smoke.cpp` pipeline.
- Link protobuf + libwebp, generate C++ protobuf sources in CMake.
- Locate the aligned 112×112 GPU crop buffer and wire it into a bounded
  per-track representative crop selector.
- Write raw `observations.pb`, `track_templates.pb`, `crops/*.webp`, and
  `manifest.json` to a temp subdir, validate, compute SHA-256/size, then atomically
  publish.

### [✓] Task 6: Implement native artifact/template/crop readers in Python
- `backend/app/infrastructure/serialization/video_track_template_reader.py`.
- `backend/app/infrastructure/serialization/native_bundle_reader.py`.
- `backend/app/infrastructure/runtime/native_representative_crop_provider.py`.
- Unit tests covering corrupt zstd/protobuf, missing crop, wrong SHA, invalid
  dimensions, mapping mismatch.

### [•] Task 7: Implement Python worker orchestration
- `backend/app/worker/video_worker_main.py`: claim loop, native subprocess,
  artifact reading, service invocation.
- `backend/app/worker/bootstrap.py`: wire `VideoProcessingService` with native
  crop provider.
- Add integration test with a fake native bundle producer; assert `completed`
  state.

### [ ] Task 8: Build pinned native worker Docker image
- Create: `backend/Dockerfile.worker.gpu` from
  `mergenvision/deepstream-dev:9.0` with protobuf + libwebp + Python deps.
- Makefile target: `make phase2-m6-native-worker-build`.

### [ ] Task 9: Wire worker services into `docker-compose.gpu.yml`
- Add `worker-gpu-0`, `worker-gpu-1`, `worker-gpu-2` pinned to distinct GPUs.
- Pass env vars: model profile, engine paths, worker lease, DB/MinIO/Qdrant URLs.

### [ ] Task 10: Add Makefile gates
- `phase2-m6-native-artifact-contract`
- `phase2-m6-native-representative-crops`
- `phase2-m6-native-exit-cleanup`
- `phase2-m7-python-worker`
- `phase2-m7-video-worker-e2e`
- `phase2-m7-video-worker-failure`
- `phase2-video-backend-e2e`

### [ ] Task 11: Run real Friends GPU worker E2E
- Upload `Friends.mp4`.
- Assert native worker produces observations/templates/crops.
- Assert job reaches `completed` and result API returns people.

### [ ] Task 12: Add retry/cancel/failure/cleanup integration tests
- Fake native bundle producer success/failure.
- 10 short native runs with deterministic output and no stale PIDs/containers/GPU
  memory growth.

---

## Acceptance Criteria

1. `make phase2-step0-static` passes and finds no forbidden strings in production
   worker paths.
2. `make phase2-m6-native-worker-build` produces a working `mv_video_worker` binary.
3. Running `mv_video_worker` on `Friends.mp4` produces:
   - `observations.pb.zst` with frame/detection records,
   - `track_templates.pb.zst` with 512-D templates,
   - `crops/*.webp` with 112×112 valid aligned crops,
   - `manifest.json` with consistent counts and artifact SHA-256/size entries.
4. Python observation + template + bundle readers load the artifacts and reject
   corrupt inputs with structured errors.
5. `make phase2-m7-video-worker-e2e` uploads `Friends.mp4`, the job reaches
   `completed`, and the result API returns people.
6. The native process naturally exits and no stale containers / GPU allocations
   remain after the run.

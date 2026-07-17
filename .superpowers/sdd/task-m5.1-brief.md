# Task M5.1 — Sequential Frame/Batch Contract Unit Tests

## Where this fits

This is the first task of the M5.1/M5.2/M6 native video-worker build. It is a CPU-only metadata/TDD task that must run *before* any GPU gate. The output is a deterministic, tested native contract for frame identity, temporal batch assembly, detection mapping, recognition mapping, and tracker batch-boundary behavior.

## Global constraints (verbatim)

- Detector model: `retinaface_r50_dynamic.onnx`
- Recognizer model: `glintr100.onnx`
- DeepStream/GStreamer/NVDEC; CUDA/TensorRT hot path
- No CPU decode / CPU inference fallback
- No OpenCV/PIL/software decode
- `research/video_reference_lab/**` is frozen; do not touch
- No annotated MP4 / NVENC
- No embedding written to PostgreSQL
- UUIDv7 for all persistent opaque IDs
- No git add/commit/push
- No `docker compose down -v`
- No placeholders. No duplicate implementations. No host-only contract may be reported as native runtime PASS.

## Required deliverables

1. Frame identity contract header/struct in `backend/native/video_worker/`:
   - `presentation_index` (canonical zero-based video frame number, assigned before any sampling)
   - `decoded_sequence` (raw decoder order counter)
   - `sampled_sequence` (counter of frames actually sent to inference)
   - `mux_batch_sequence` (GstBuffer / nvstreammux batch counter)
   - `position_in_mux_batch` (`NvDsFrameMeta.batch_id` surface slot, NOT frame number)
   - `inference_batch_sequence` (RetinaFace TensorRT batch counter)
   - `position_in_inference_batch` (index inside that TensorRT batch, `0..B-1`)
   - `source_id`, `pad_index`, `nvds_frame_num`
   - `pts_ns`, `duration_ns`, `pts_derived`
   - source/display dimensions, rotation
   - `DeviceImageView`, `RetainedBufferHandle` (ownership abstraction)

2. Retained buffer ownership abstraction:
   - Retains source `GstSample`/`GstBuffer` ref until the batch's inference + GPU work is complete.
   - Fake retained handle usable in unit tests to prove lifetime rules.
   - No frame-level `cudaDeviceSynchronize()`.

3. `TemporalFrameBatchAssembler`:
   - Constructor takes `max_batch_size` (default 8).
   - `push(std::vector<FrameEnvelope>)` returns complete `InferenceFrameBatch` objects.
   - `flush_eos()` returns the final partial batch.
   - `cancel()` releases everything.
   - Invariants: frames ascend by `presentation_index` and `pts_ns`; no duplicates/gaps (when sampling disabled); size `1..max_batch_size`; `position_in_inference_batch` recomputed inside each assembled batch.

4. Mapping helpers:
   - Frame-detections mapping: input batch size == output `FrameDetections` size; preserves identity even if TensorRT output order is shuffled.
   - Recognition crop mapping: deterministic ordering `(presentation_index, detection_ordinal)`; GlintR100 chunking up to batch size 32; embeddings reattached by exact crop ref index.

5. Tracker batch-boundary test support:
   - Provide a minimal deterministic job-local track allocator (`RT000001`, ...).
   - Provide a minimal Python/C++ tracker adapter that can be driven chronologically and preserves track identity across detector batch boundaries.

## Exact test cases the target must cover

> Target name in Makefile: `phase2-m5-sequence-contract`.
> The target must run real tests. Empty collection or skip-only is a failure.

A. Sequential assembler: input F0..F17, max_batch=8 → `[F0..F7]`, `[F8..F15]`, `[F16..F17]`.
B. Irregular mux buffers: chunks `[F0,F1,F2]`, `[F3]`, `[F4,F5,F6,F7,F8]`, `[F9]` → inference batches `[F0..F7]`, `[F8,F9]`.
C. PTS ordering inside one mux buffer: input `[F2, F0, F1]` → canonical order `[F0, F1, F2]`.
D. PTS regression: previous emitted PTS 100ms, next frame PTS 80ns → raise `VIDEO_PRESENTATION_ORDER_VIOLATION` and abort.
E. Sampling: decoded F0..F9, every_n=3 → processed presentation indices `[0,3,6,9]`, sampled_sequence `[0,1,2,3]`.
F. Owner lifetime: fake retained handles prove batch is not released before inference completion; EOS/error/cancel releases all refs.
G. Detector mapping: shuffled TensorRT output still maps to correct `FrameIdentity` by `position_in_inference_batch`.
H. Recognition mapping: 33 eligible crops → two Glint batches (32+1); embeddings reattached to exact `presentation_index`/`detection_ordinal`.
I. Track batch boundary: same object visible in frames 6,7 and 8,9 with batch boundary at 7/8 → same `local_track_key`.
J. Chunk invariance: same synthetic data run with detector batch sizes 1, 4, 8 → identical logical frame/detection/track mapping.

## Interaction with existing source

Existing source to extend:
- `backend/native/video_worker/CMakeLists.txt`
- `backend/native/video_worker/tests/decode_smoke.cpp`
- Existing `backend/native/image_runtime/` FacePipeline and CUDA/TensorRT kernels may be reused/adapted, but no duplicate pipeline created.

Do not create a second implementation under a different name. Extend what exists.

## Report contract

Write a full report to `backend/native/video_worker/.superpowers/sdd/task-m5.1-report.md` (create parent dirs as needed) containing:
- status: DONE / DONE_WITH_CONCERNS / BLOCKED
- files created/changed
- test command run and raw output
- any skipped tests and why
- concerns/blockers

"""GPU batch extraction pipeline wrapper."""

from __future__ import annotations

import contextlib
import ctypes
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
from cuda.bindings import runtime as cuda_runtime

from mv_phase1_bulk.alignment import GpuFaceAligner
from mv_phase1_bulk.buffer_arena import BufferArena
from mv_phase1_bulk.decoder import DecodeInfo, JpegGpuDecoder
from mv_phase1_bulk.device_tensor import DeviceTensor, check_cuda
from mv_phase1_bulk.recognizer import GpuRecognizer
from mv_phase1_bulk.retinaface_postprocess import RetinaFacePostprocess
from mv_phase1_bulk.retinaface_preprocessor import RetinaFacePreprocessor
from mv_phase1_bulk.trt_device_engine import TrtDeviceEngine
from mv_phase1_bulk.types import FaceExtraction, ImageExtractionResult

logger = logging.getLogger(__name__)

# One pipeline stream per device is reused across pipeline lifetimes.  Destroying
# the handle while CV-CUDA may cache the underlying stream value causes
# cudaErrorInvalidResourceHandle on the next pipeline creation in the same process.
_SHARED_STREAMS: dict[int, int] = {}


class GpuFacePipeline:
    """Single-GPU face detection + alignment + recognition pipeline."""

    def __init__(
        self,
        *,
        model_profile: dict[str, Any],
        device_id: int = 0,
    ) -> None:
        self._profile = model_profile
        self._device_id = int(device_id)
        self._model_version = model_profile["model_version"]
        self._preprocess_version = model_profile["preprocess_version"]

        det_cfg = model_profile["detector"]
        rec_cfg = model_profile["recognizer"]
        eng_cfg = model_profile["engine_manifest"]

        self._detector_input_size = int(det_cfg["input_shape"][2])
        self._detector_conf_threshold = float(det_cfg["confidence_threshold"])
        self._detector_nms_threshold = float(det_cfg["nms_threshold"])
        self._embedding_dim = int(rec_cfg["embedding_dim"])

        self._set_device()
        self._decoder = JpegGpuDecoder(device_id=device_id)
        self._preprocessor = RetinaFacePreprocessor(
            input_size=self._detector_input_size,
            device_id=device_id,
        )
        self._detector_engine = TrtDeviceEngine(
            Path(eng_cfg["retinaface_r50_dynamic"]["engine_path"]),
            device_id=device_id,
        )
        self._postprocess = RetinaFacePostprocess(
            input_size=self._detector_input_size,
            device_id=device_id,
            max_candidates=int(det_cfg.get("max_candidates", 2000)),
        )
        self._aligner = GpuFaceAligner(device_id=device_id)
        self._recognizer = GpuRecognizer(
            engine_path=Path(eng_cfg["glintr100"]["engine_path"]),
            device_id=device_id,
            embedding_dim=self._embedding_dim,
        )
        self._arena = BufferArena(device_id=device_id)
        self._set_device()
        if device_id not in _SHARED_STREAMS:
            err, stream = cuda_runtime.cudaStreamCreate()
            check_cuda(err, "pipeline stream create")
            _SHARED_STREAMS[device_id] = stream
        self._stream = _SHARED_STREAMS[device_id]

    def _set_device(self) -> None:
        err = cuda_runtime.cudaSetDevice(self._device_id)
        check_cuda(err, f"cudaSetDevice({self._device_id})")

    def warmup(self) -> None:
        self._set_device()
        input_name = self._detector_engine._input_names[0]
        self._detector_engine.warmup({input_name: (1, 3, self._detector_input_size, self._detector_input_size)})
        self._recognizer.warmup()
        logger.info("GPU face pipeline warmup complete")

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        with contextlib.suppress(Exception):
            cuda_runtime.cudaSetDevice(self._device_id)
        if hasattr(self, "_stream"):
            with contextlib.suppress(Exception):
                cuda_runtime.cudaStreamSynchronize(self._stream)
        for obj in (
            self._decoder,
            self._preprocessor,
            self._postprocess,
            self._aligner,
            self._recognizer,
            self._detector_engine,
            self._arena,
        ):
            close_method = getattr(obj, "close", None)
            if callable(close_method):
                with contextlib.suppress(Exception):
                    close_method()
        # Drop references to CUDA-heavy objects while the context is still valid.
        # This avoids destructors running during interpreter teardown with a stale
        # context / stream handle.
        self._decoder = cast(Any, None)
        self._preprocessor = cast(Any, None)
        self._detector_engine = cast(Any, None)
        self._postprocess = cast(Any, None)
        self._aligner = cast(Any, None)
        self._recognizer = cast(Any, None)
        self._arena = cast(Any, None)
        # The shared stream is intentionally kept alive; see _SHARED_STREAMS note.

    def __del__(self) -> None:
        if getattr(self, "_closed", False):
            return
        with contextlib.suppress(Exception):
            self.close()

    def __enter__(self) -> GpuFacePipeline:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def extract_batch(
        self,
        image_bytes_list: list[bytes],
        *,
        source_keys: list[str] | None = None,
        max_batch: int = 256,
        multi_face_policy: str = "quarantine",
    ) -> list[ImageExtractionResult]:
        """Batch extraction optimised for RetinaFace R50 dynamic batch.

        Parameters
        ----------
        image_bytes_list:
            Encoded JPEG bytes for each image.
        source_keys:
            Optional stable identifier for each image; defaults to indices.
        max_batch:
            Maximum number of images processed in one detector/recognizer call.
        multi_face_policy:
            ``quarantine`` (default) rejects images with more than one detected
            face; ``largest`` keeps the largest face per image.

        Returns
        -------
        One ``ImageExtractionResult`` per input, preserving input order.
        """
        self._set_device()
        n = len(image_bytes_list)
        if source_keys is None:
            source_keys = [str(i) for i in range(n)]
        if len(source_keys) != n:
            raise ValueError("source_keys length must match image_bytes_list")

        if n == 0:
            return []

        results: list[ImageExtractionResult] = [
            ImageExtractionResult(
                source_index=i,
                source_key=source_keys[i],
                original_width=0,
                original_height=0,
                status="pending",
                rejection_reason=None,
                faces=[],
            )
            for i in range(n)
        ]

        batch_size = min(n, max_batch)
        for chunk_start in range(0, n, batch_size):
            chunk = image_bytes_list[chunk_start : chunk_start + batch_size]
            chunk_keys = source_keys[chunk_start : chunk_start + batch_size]
            try:
                self._process_chunk(
                    chunk,
                    chunk_keys,
                    chunk_start,
                    results,
                    multi_face_policy=multi_face_policy,
                )
            except Exception as exc:
                logger.exception("chunk %d failed: %s", chunk_start, exc)
                for i, _key in enumerate(chunk_keys):
                    res = results[chunk_start + i]
                    res.status = "inference-error"
                    res.rejection_reason = f"chunk_error: {type(exc).__name__}: {exc}"

        return results

    def _process_chunk(
        self,
        chunk: list[bytes],
        chunk_keys: list[str],
        chunk_start: int,
        results: list[ImageExtractionResult],
        *,
        multi_face_policy: str,
    ) -> None:
        b = len(chunk)

        # Decode
        d_images: list[DeviceTensor | None] = []
        infos: list[DecodeInfo | None] = []
        try:
            decoded, decoded_infos = self._decoder.decode_batch(chunk, stream=int(self._stream))
            d_images = list(decoded)
            infos = list(decoded_infos)
        except Exception as exc:
            logger.warning("batch decode failed, falling back to per-image: %s", exc)
            for encoded in chunk:
                try:
                    d_img, decoded_info = self._decoder.decode(encoded, stream=int(self._stream))
                    d_images.append(d_img)
                    infos.append(decoded_info)
                except Exception as inner:
                    d_images.append(None)
                    infos.append(None)
                    logger.debug("single decode failed: %s", inner)

        # Preprocess only successfully decoded images, preserving original index
        # so detection outputs can be aligned with per-image metadata.
        valid_indices: list[int] = [idx for idx, img in enumerate(d_images) if img is not None]
        valid_images: list[DeviceTensor] = [cast(DeviceTensor, d_images[idx]) for idx in valid_indices]
        valid_infos: list[DecodeInfo] = [cast(DecodeInfo, infos[idx]) for idx in valid_indices]
        if not valid_images or len(valid_images) != len(valid_infos):
            for i in range(b):
                res = results[chunk_start + i]
                res.status = "decode_failed"
                res.rejection_reason = "jpeg_decode_failed"
            return

        d_input = self._preprocessor.preprocess_batch(valid_images, stream=int(self._stream))

        # Detect
        input_name = self._detector_engine._input_names[0]
        det_outputs = self._detector_engine.infer_device(
            {input_name: d_input},
            stream=int(self._stream),
        )
        per_image = self._postprocess.decode(
            det_outputs,
            conf_threshold=self._detector_conf_threshold,
            nms_threshold=self._detector_nms_threshold,
            stream=int(self._stream),
        )
        scaled_list = self._postprocess.scale_and_compact(
            per_image,
            original_heights=[info.height for info in valid_infos],
            original_widths=[info.width for info in valid_infos],
            stream=int(self._stream),
        )

        # Per-image face counts drive both the quarantine decision and the
        # largest-face selection.  Copy them back explicitly before the kernel
        # so the host policy logic and the native picker see consistent data.
        face_counts_host = np.empty(len(scaled_list), dtype=np.int32)
        for i, scaled in enumerate(scaled_list):
            err = cuda_runtime.cudaMemcpyAsync(
                int(face_counts_host.ctypes.data) + i * 4,
                scaled.count.ptr,
                4,
                cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                int(self._stream),
            )
            check_cuda(err, "face count D2H")
        err = cuda_runtime.cudaStreamSynchronize(int(self._stream))
        check_cuda(err, "face count sync")

        # Per-image largest-face selection stays on device.
        batch_selections = self._postprocess.pick_largest_device(
            scaled_list,
            stream=int(self._stream),
        )
        valid_host = np.empty(len(valid_images), dtype=np.int32)
        err = cuda_runtime.cudaMemcpyAsync(
            valid_host.ctypes.data,
            batch_selections.valid.ptr,
            valid_host.nbytes,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            int(self._stream),
        )
        check_cuda(err, "batch selections valid D2H")
        err = cuda_runtime.cudaStreamSynchronize(int(self._stream))
        check_cuda(err, "valid sync")

        # Pull all selected detection metadata to the host in one set of async
        # copies and synchronize once before any host-side read.  This removes
        # the prior race where score_h[0] was read before the stream completed.
        num_valid_detections = len(scaled_list)
        boxes_h = np.empty((num_valid_detections, 4), dtype=np.float32)
        landmarks_h = np.empty((num_valid_detections, 10), dtype=np.float32)
        scores_h = np.empty((num_valid_detections,), dtype=np.float32)
        if num_valid_detections:
            err = cuda_runtime.cudaMemcpyAsync(
                boxes_h.ctypes.data,
                batch_selections.boxes.ptr,
                boxes_h.nbytes,
                cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                int(self._stream),
            )
            check_cuda(err, "selected boxes D2H")
            err = cuda_runtime.cudaMemcpyAsync(
                landmarks_h.ctypes.data,
                batch_selections.landmarks.ptr,
                landmarks_h.nbytes,
                cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                int(self._stream),
            )
            check_cuda(err, "selected landmarks D2H")
            err = cuda_runtime.cudaMemcpyAsync(
                scores_h.ctypes.data,
                batch_selections.scores.ptr,
                scores_h.nbytes,
                cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                int(self._stream),
            )
            check_cuda(err, "selected scores D2H")
            err = cuda_runtime.cudaStreamSynchronize(int(self._stream))
            check_cuda(err, "selected rows sync")

        # Build compact selected metadata in host order.
        selected: list[tuple[int, int]] = []
        valid_idx = 0
        for i in range(b):
            res = results[chunk_start + i]
            info = infos[i]
            if info is None:
                res.status = "decode_failed"
                res.rejection_reason = "jpeg_decode_failed"
                continue
            res.original_width = info.width
            res.original_height = info.height

            face_count = int(face_counts_host[valid_idx])
            if face_count == 0 or not valid_host[valid_idx]:
                res.status = "no_face"
                res.rejection_reason = "no_face_detected"
                valid_idx += 1
                continue

            if multi_face_policy == "quarantine" and face_count > 1:
                res.status = "quarantine"
                res.rejection_reason = "multiple_faces"
                valid_idx += 1
                continue

            selected.append((chunk_start + i, valid_idx))
            valid_idx += 1

        if not selected:
            return

        # Batch alignment: collect all selected chips into one recognizer batch.
        m = len(selected)
        chip_batch = self._arena.reserve(
            (m, 3, 112, 112),
            ctypes.c_float,
            stream=int(self._stream),
        )
        chip_plane_bytes = 3 * 112 * 112 * 4

        for j, (_, valid_idx) in enumerate(selected):
            d_landmarks = DeviceTensor(
                batch_selections.landmarks.ptr + valid_idx * 10 * 4,
                (1, 10),
                ctypes.c_float,
                self._device_id,
                batch_selections.landmarks,
                stream=int(self._stream),
            )
            chip = self._aligner.align(valid_images[valid_idx], d_landmarks, stream=int(self._stream))
            err = cuda_runtime.cudaMemcpyAsync(
                chip_batch.ptr + j * chip_plane_bytes,
                chip.ptr,
                chip_plane_bytes,
                cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                int(self._stream),
            )
            check_cuda(err, "extract_batch chip D2D")

        d_embeddings = self._recognizer.embed(
            DeviceTensor(
                ptr=chip_batch.ptr,
                shape=(m, 3, 112, 112),
                dtype=ctypes.c_float,
                device_id=self._device_id,
                owner=chip_batch,
                stream=int(self._stream),
            ),
            stream=int(self._stream),
        )

        embeddings_host = np.empty(d_embeddings.shape, dtype=np.float32)
        err = cuda_runtime.cudaMemcpyAsync(
            embeddings_host.ctypes.data,
            d_embeddings.ptr,
            embeddings_host.nbytes,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            int(self._stream),
        )
        check_cuda(err, "extract_batch embeddings D2H")

        # Wait for the embedding D2H before the host reads it.
        err = cuda_runtime.cudaStreamSynchronize(int(self._stream))
        check_cuda(err, "extract_batch final sync")

        for j, (global_idx, valid_idx) in enumerate(selected):
            res = results[global_idx]
            embedding = embeddings_host[j]
            norm = float(np.linalg.norm(embedding))
            face = FaceExtraction(
                source_index=global_idx,
                detection_ordinal=0,
                bbox_original=boxes_h[valid_idx],
                landmarks_original=landmarks_h[valid_idx].reshape(5, 2),
                detector_score=float(scores_h[valid_idx]),
                quality_primitives={"l2_norm": norm},
                embedding=embedding,
                embedding_norm=norm,
                crop_bytes=b"",  # aligned crop is not persisted; original bytes stored separately
                model_version=self._model_version,
                preprocess_version=self._preprocess_version,
            )
            res.faces.append(face)
            res.status = "accepted"

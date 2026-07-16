"""Official InsightFace FaceAnalysis oracle for reference extraction.

This oracle uses ``insightface.app.FaceAnalysis`` with the official
``buffalo_l`` pack. It does not perform custom decoding; all preprocess,
alignment, and embedding logic is delegated to InsightFace so that the
reference output matches the official implementation.

Normal imports and runs must not download models. Use
``mv-video-lab models acquire --name buffalo_l --allow-download`` to populate
the cache first.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from mergenvision_video_lab.config import resolve_repo_relative_path
from mergenvision_video_lab.contracts import Landmarks5, OnnxModelContract
from mergenvision_video_lab.errors import ModelArtifactError
from mergenvision_video_lab.geometry import clamp_bbox
from mergenvision_video_lab.hashing import sha256_file
from mergenvision_video_lab.model_inventory import available_providers, select_providers
from mergenvision_video_lab.quality import compute_quality


@dataclass(frozen=True, slots=True)
class OracleDetection:
    """One face detection from the oracle, before tracking/quality filtering."""

    bbox_xyxy: tuple[float, float, float, float]
    detector_score: float
    landmarks_5: Landmarks5
    aligned_crop: np.ndarray
    embedding: np.ndarray | None
    quality: dict[str, Any]


class FaceAnalysisOracle:
    """Detector/recognizer oracle backed by official InsightFace FaceAnalysis."""

    def __init__(
        self,
        model_root: str | None,
        model_pack: str | None,
        requested_provider: Literal["cuda", "cpu"],
        allow_cpu_fallback: bool,
        det_size: Sequence[int] = (640, 640),
    ) -> None:
        if model_pack is None:
            raise ModelArtifactError("model_pack must be configured (e.g. buffalo_l)")
        if model_pack != "buffalo_l":
            raise ModelArtifactError(
                f"unsupported model pack: {model_pack}",
                {"supported": ["buffalo_l"]},
            )

        self.model_pack = model_pack
        self.det_size = (int(det_size[0]), int(det_size[1]))
        self.requested_provider = requested_provider
        self.allow_cpu_fallback = allow_cpu_fallback

        self._available_providers: list[str] = available_providers()
        self._actual_providers: list[str] = select_providers(
            requested_provider,
            self._available_providers,
            allow_cpu_fallback,
        )

        # Map provider to InsightFace ctx_id.
        if "CUDAExecutionProvider" in self._actual_providers:
            self._ctx_id = 0
        else:
            self._ctx_id = -1

        self._root = self._resolve_root(model_root)
        self._pack_dir = self._root / "models" / model_pack
        if not self._pack_dir.exists():
            raise ModelArtifactError(
                "model pack not found; run 'mv-video-lab models acquire' first",
                {"pack_dir": str(self._pack_dir)},
            )

        self._app = self._build_app()
        self.detector_contract = self._build_contract("det_10g.onnx", "detection")
        self.recognizer_contract = self._build_contract("w600k_r50.onnx", "recognition")

    def _resolve_root(self, model_root: str | None) -> Path:
        """Resolve the InsightFace root directory from config."""
        if model_root is None:
            return Path.home() / ".insightface"
        path = resolve_repo_relative_path(model_root)
        if path.is_absolute():
            return path
        return Path.cwd() / path

    def _build_app(self) -> Any:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ModelArtifactError(f"insightface not installed: {exc}") from exc

        app = FaceAnalysis(
            name=self.model_pack,
            root=str(self._root),
            allowed_modules=["detection", "recognition"],
        )
        # Use a low detection threshold; the lab applies its own thresholds later.
        app.prepare(ctx_id=self._ctx_id, det_thresh=0.1, det_size=self.det_size)
        return app

    def _build_contract(self, basename: str, task: str) -> OnnxModelContract:
        path = self._pack_dir / basename
        if not path.exists():
            raise ModelArtifactError(
                f"missing {task} model file",
                {"expected": str(path)},
            )
        try:
            import onnx

            model = onnx.load(str(path))
        except Exception as exc:
            raise ModelArtifactError(f"cannot load ONNX {basename}: {exc}") from exc

        inputs = []
        for inp in model.graph.input:
            tensor_type = inp.type.tensor_type
            shape = [
                dim.dim_value if dim.dim_value > 0 else dim.dim_param or "dynamic"
                for dim in tensor_type.shape.dim
            ]
            inputs.append(
                {
                    "name": inp.name,
                    "shape": shape,
                    "dtype": onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type).name,
                }
            )
        outputs = []
        for out in model.graph.output:
            tensor_type = out.type.tensor_type
            shape = [
                dim.dim_value if dim.dim_value > 0 else dim.dim_param or "dynamic"
                for dim in tensor_type.shape.dim
            ]
            outputs.append(
                {
                    "name": out.name,
                    "shape": shape,
                    "dtype": onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type).name,
                }
            )
        return OnnxModelContract(
            basename=path.name,
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
            inputs=inputs,
            outputs=outputs,
            opset=model.opset_import[0].version if model.opset_import else None,
            producer=model.producer_name or None,
        )

    def detect(
        self,
        image: np.ndarray,
        detector_low_threshold: float,
        frame_width: int,
        frame_height: int,
        quality_config: dict[str, Any],
        compute_embeddings: bool = True,
    ) -> list[OracleDetection]:
        """Run detector on a BGR image and return detections."""
        faces = self._app.get(image, max_num=0)
        results: list[OracleDetection] = []
        for face in faces:
            score = float(face.det_score)
            if score < detector_low_threshold:
                continue

            bbox = face.bbox.astype(np.float32)
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, frame_width, frame_height)
            if x2 <= x1 or y2 <= y1:
                continue

            kps = face.kps.astype(np.float32)
            landmarks = Landmarks5(
                left_eye=tuple(kps[0]),
                right_eye=tuple(kps[1]),
                nose=tuple(kps[2]),
                left_mouth=tuple(kps[3]),
                right_mouth=tuple(kps[4]),
            )

            # Use the official upstream alignment for the reference crop.
            from insightface.utils.face_align import norm_crop

            aligned_crop = norm_crop(image, kps, image_size=112, mode="arcface")

            embedding: np.ndarray | None = None
            if compute_embeddings and hasattr(face, "embedding"):
                emb = face.embedding
                if emb is not None and emb.size > 0:
                    embedding = np.asarray(emb, dtype=np.float32).flatten()
                    norm = float(np.linalg.norm(embedding))
                    if norm > 0 and np.isfinite(norm):
                        embedding = (embedding / norm).astype(np.float32)
                    else:
                        embedding = None

            quality = compute_quality(
                aligned_crop=aligned_crop,
                bbox_xyxy=(x1, y1, x2, y2),
                frame_width=frame_width,
                frame_height=frame_height,
                landmarks_5=landmarks.to_array(),
                detector_score=score,
                reprojection_error_px=0.0,  # upstream alignment residual not exposed
                config=quality_config,
            )

            results.append(
                OracleDetection(
                    bbox_xyxy=(x1, y1, x2, y2),
                    detector_score=score,
                    landmarks_5=landmarks,
                    aligned_crop=aligned_crop,
                    embedding=embedding,
                    quality=quality.model_dump(mode="json"),
                )
            )

        # Deterministic ordering: x1, y1, x2, y2, score descending.
        results.sort(
            key=lambda d: (d.bbox_xyxy[0], d.bbox_xyxy[1], d.bbox_xyxy[2], -d.detector_score)
        )
        return results


# Keep the old name for callers that import InsightFaceOracle.
InsightFaceOracle = FaceAnalysisOracle

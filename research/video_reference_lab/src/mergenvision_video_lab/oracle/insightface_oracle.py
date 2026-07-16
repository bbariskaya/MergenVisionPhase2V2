"""ONNX RetinaFace detector + ArcFace recognizer with InsightFace-aligned parity.

This oracle loads the local ONNX artifacts directly via ONNX Runtime rather than
using ``insightface.app.FaceAnalysis``, because the provided models are loose
ONNX files. Alignment parity against ``insightface.utils.face_align.norm_crop``
is enforced by unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import cv2
import numpy as np
import onnxruntime as ort

from mergenvision_video_lab.alignment import align_face
from mergenvision_video_lab.contracts import BBoxXYXY, Landmarks5
from mergenvision_video_lab.geometry import clamp_bbox
from mergenvision_video_lab.model_inventory import ModelInventory, select_providers
from mergenvision_video_lab.quality import compute_quality


@dataclass(frozen=True, slots=True)
class OracleDetection:
    """One face detection from the oracle, before tracking/quality filtering."""

    bbox_xyxy: tuple[float, float, float, float]
    detector_score: float
    landmarks_5: Landmarks5
    aligned_crop: np.ndarray
    embedding: np.ndarray | None
    quality: dict


# Standard RetinaFace anchor configuration for 640x640.
_RETINA_MIN_SIZES = [[16, 32], [64, 128], [256, 512]]
_RETINA_STEPS = [8, 16, 32]
_RETINA_VARIANCE = [0.1, 0.2]
_RETINA_IMAGE_SIZE = 640


def _generate_anchors(
    image_size: int = _RETINA_IMAGE_SIZE,
    min_sizes: list[list[int]] = _RETINA_MIN_SIZES,
    steps: list[int] = _RETINA_STEPS,
) -> np.ndarray:
    """Generate normalized RetinaFace anchors (cx, cy, w, h)."""
    anchors = []
    for step, sizes in zip(steps, min_sizes):
        feature_size = image_size // step
        for y in range(feature_size):
            for x in range(feature_size):
                for min_size in sizes:
                    cx = (x + 0.5) * step / image_size
                    cy = (y + 0.5) * step / image_size
                    w = min_size / image_size
                    h = min_size / image_size
                    anchors.extend([cx, cy, w, h])
    return np.array(anchors, dtype=np.float32).reshape(-1, 4)


def _decode_bboxes(loc: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Decode RetinaFace bbox regressions to XYXY in normalized coords."""
    anchors_xy = anchors[:, :2]
    anchors_wh = anchors[:, 2:]
    boxes_xy = anchors_xy + loc[:, :2] * _RETINA_VARIANCE[0] * anchors_wh
    boxes_wh = anchors_wh * np.exp(loc[:, 2:] * _RETINA_VARIANCE[1])
    x1 = boxes_xy[:, 0] - boxes_wh[:, 0] / 2
    y1 = boxes_xy[:, 1] - boxes_wh[:, 1] / 2
    x2 = boxes_xy[:, 0] + boxes_wh[:, 0] / 2
    y2 = boxes_xy[:, 1] + boxes_wh[:, 1] / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def _decode_landmarks(landms: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Decode RetinaFace landmarks to normalized (x,y) x5."""
    anchors_xy = anchors[:, :2]
    anchors_wh = anchors[:, 2:]
    landms = landms.reshape(-1, 5, 2)
    pts = anchors_xy[:, None, :] + landms * _RETINA_VARIANCE[0] * anchors_wh[:, None, :]
    return pts.reshape(-1, 5, 2)


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """Greedy IoU NMS returning surviving indices."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= threshold]
    return keep


def _detector_preprocessing(image: np.ndarray, det_size: tuple[int, int]) -> np.ndarray:
    """Resize and normalize image for RetinaFace detector."""
    resized = cv2.resize(image, det_size)
    tensor = resized.astype(np.float32)
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    return tensor


def _recognizer_preprocessing(aligned_crop: np.ndarray) -> np.ndarray:
    """Normalize aligned 112x112 crop for ArcFace recognizer."""
    tensor = aligned_crop.astype(np.float32) / 255.0
    tensor = (tensor - 0.5) / 0.5
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    return tensor


class InsightFaceOracle:
    """Offline detector/recognizer oracle backed by local ONNX artifacts."""

    def __init__(
        self,
        model_root: str | None,
        model_pack: str | None,
        requested_provider: Literal["cuda", "cpu"],
        allow_cpu_fallback: bool,
        det_size: Sequence[int] = (640, 640),
    ) -> None:
        self.det_size = tuple(det_size)
        self._inventory = ModelInventory(model_root, model_pack)
        self._available_providers = ort.get_available_providers()
        self._actual_providers = select_providers(
            requested_provider,
            self._available_providers,
            allow_cpu_fallback,
        )
        self._det_session = ort.InferenceSession(
            str(self._inventory.detector_path),
            providers=self._actual_providers,
        )
        self._rec_session = ort.InferenceSession(
            str(self._inventory.recognizer_path),
            providers=self._actual_providers,
        )
        self._anchors = _generate_anchors(
            image_size=self.det_size[0],
        )

    @property
    def detector_contract(self):
        return self._inventory.detector_contract

    @property
    def recognizer_contract(self):
        return self._inventory.recognizer_contract

    def detect(
        self,
        image: np.ndarray,
        detector_low_threshold: float,
        frame_width: int,
        frame_height: int,
        quality_config: dict,
        compute_embeddings: bool = True,
    ) -> list[OracleDetection]:
        """Run detector on a BGR image and return detections."""
        tensor = _detector_preprocessing(image, self.det_size)
        loc, conf, landms = self._det_session.run(None, {"input": tensor})

        loc = loc[0]
        conf = conf[0]
        landms = landms[0]

        scores = conf[:, 1]
        valid_mask = scores >= detector_low_threshold
        if not np.any(valid_mask):
            return []

        loc = loc[valid_mask]
        landms = landms[valid_mask]
        scores = scores[valid_mask]
        anchors = self._anchors[valid_mask]

        boxes = _decode_bboxes(loc, anchors)
        pts = _decode_landmarks(landms, anchors)

        # Scale from detector input space to original image space.
        scale_x = frame_width / self.det_size[0]
        scale_y = frame_height / self.det_size[1]
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        pts[:, :, 0] *= scale_x
        pts[:, :, 1] *= scale_y

        keep = _nms(boxes, scores, threshold=0.4)

        results: list[OracleDetection] = []
        for idx in keep:
            x1, y1, x2, y2 = boxes[idx]
            x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, frame_width, frame_height)
            if x2 <= x1 or y2 <= y1:
                continue

            lm = pts[idx]
            landmarks = Landmarks5(
                left_eye=tuple(lm[0]),
                right_eye=tuple(lm[1]),
                nose=tuple(lm[2]),
                left_mouth=tuple(lm[3]),
                right_mouth=tuple(lm[4]),
            )

            try:
                aligned_crop, matrix, reproj_error = align_face(
                    image,
                    landmarks.to_array(),
                    output_size=112,
                    color_order="BGR",
                )
            except Exception:
                continue

            embedding: np.ndarray | None = None
            if compute_embeddings:
                rec_input = _recognizer_preprocessing(aligned_crop)
                embedding = self._rec_session.run(None, {"input.1": rec_input})[0][0]
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

            quality = compute_quality(
                aligned_crop=aligned_crop,
                bbox_xyxy=(x1, y1, x2, y2),
                frame_width=frame_width,
                frame_height=frame_height,
                landmarks_5=landmarks.to_array(),
                detector_score=float(scores[idx]),
                reprojection_error_px=reproj_error,
                config=quality_config,
            )

            results.append(
                OracleDetection(
                    bbox_xyxy=(x1, y1, x2, y2),
                    detector_score=float(scores[idx]),
                    landmarks_5=landmarks,
                    aligned_crop=aligned_crop,
                    embedding=embedding,
                    quality=quality.model_dump(mode="json"),
                )
            )

        # Deterministic ordering: x1, y1, x2, y2, score descending.
        results.sort(key=lambda d: (d.bbox_xyxy[0], d.bbox_xyxy[1], d.bbox_xyxy[2], -d.detector_score))
        return results

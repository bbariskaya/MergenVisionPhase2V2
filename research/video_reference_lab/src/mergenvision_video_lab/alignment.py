"""ArcFace five-point alignment to 112x112 canonical template."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
from skimage.transform import SimilarityTransform

# ArcFace 112x112 canonical five-point template.
# Order: left_eye, right_eye, nose, left_mouth, right_mouth.
# "left" means subject-left (image-right for a front-facing subject).
ARCFACE_TEMPLATE_112: np.ndarray = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class AlignmentError(Exception):
    pass


def _validate_landmarks(landmarks_5: np.ndarray) -> None:
    if landmarks_5.shape != (5, 2):
        raise AlignmentError(f"expected landmarks shape (5, 2), got {landmarks_5.shape}")
    if not np.all(np.isfinite(landmarks_5)):
        raise AlignmentError("landmarks contain non-finite values")
    eyes = landmarks_5[0] - landmarks_5[1]
    if np.linalg.norm(eyes) < 1e-3:
        raise AlignmentError("degenerate landmarks: eyes are coincident")


def _to_bgr_for_processing(
    image: np.ndarray,
    color_order: Literal["RGB", "BGR"],
) -> np.ndarray:
    """Convert input to BGR for OpenCV processing while preserving semantics."""
    if color_order == "BGR":
        return image
    if color_order == "RGB":
        return np.asarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    raise AlignmentError(f"unsupported color_order: {color_order}")


def _to_original_color_order(
    image: np.ndarray,
    color_order: Literal["RGB", "BGR"],
) -> np.ndarray:
    """Convert BGR working image back to the requested output color order."""
    if color_order == "BGR":
        return image
    if color_order == "RGB":
        return np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    raise AlignmentError(f"unsupported color_order: {color_order}")


def align_face(
    image: np.ndarray,
    landmarks_5: np.ndarray,
    output_size: int = 112,
    color_order: Literal["RGB", "BGR"] = "BGR",
    border_mode: str = "constant_zero",
    interpolation: str = "bilinear",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Align a face to the ArcFace canonical template.

    Returns:
        aligned_crop: uint8 image of shape (output_size, output_size, 3).
        transform_matrix: 2x3 affine matrix used for warp.
        reprojection_error_px: mean Euclidean reprojection error of landmarks.

    Raises:
        AlignmentError on degenerate/non-finite landmarks.
    """
    _validate_landmarks(landmarks_5)

    if image.ndim != 3 or image.shape[2] != 3:
        raise AlignmentError("image must be HxWx3 BGR/RGB")

    working = _to_bgr_for_processing(image, color_order)

    dst = ARCFACE_TEMPLATE_112 * (output_size / 112.0)
    try:
        tform = SimilarityTransform.from_estimate(landmarks_5, dst)
    except AttributeError:
        # scikit-image <0.26 fallback
        tform = SimilarityTransform()  # type: ignore[no-untyped-call]
        success = tform.estimate(landmarks_5, dst)
        if not success:
            raise AlignmentError("could not estimate similarity transform") from None

    matrix = tform.params[0:2, :].astype(np.float32)

    inter = cv2.INTER_LINEAR if interpolation == "bilinear" else cv2.INTER_NEAREST
    border = 0 if border_mode == "constant_zero" else cv2.BORDER_REPLICATE

    warped = cv2.warpAffine(
        working,
        matrix,
        (output_size, output_size),
        flags=inter,
        borderMode=border,
        borderValue=(0, 0, 0),
    )

    aligned = _to_original_color_order(warped, color_order)

    # Reprojection residual in original pixel space.
    ones = np.ones((5, 1), dtype=np.float32)
    homogeneous = np.concatenate([landmarks_5, ones], axis=1)
    projected = (matrix @ homogeneous.T).T
    reprojection_error_px = float(np.mean(np.linalg.norm(projected - dst, axis=1)))

    if not np.all(np.isfinite(aligned)):
        raise AlignmentError("aligned crop contains non-finite values")

    return aligned, matrix, reprojection_error_px

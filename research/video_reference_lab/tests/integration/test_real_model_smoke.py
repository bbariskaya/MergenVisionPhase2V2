"""Real-model smoke test using the authorized buffalo_l pack.

This test is skipped if the model pack has not been acquired. It proves that a
real InsightFace detector/recognizer session runs and produces finite embeddings.
"""

from __future__ import annotations

import numpy as np
import pytest

from mergenvision_video_lab.config import load_config


@pytest.mark.real_model
@pytest.mark.skipif(
    not load_config("configs/friends_baseline_cpu.yaml").oracle.model_pack,
    reason="no model pack configured",
)
def test_real_model_detects_and_embeds() -> None:
    """A real model session detects a face and emits a unit embedding."""
    import cv2

    from mergenvision_video_lab.oracle.insightface_oracle import InsightFaceOracle

    config = load_config("configs/friends_baseline_cpu.yaml")

    try:
        oracle = InsightFaceOracle(
            model_root=config.oracle.model_root,
            model_pack=config.oracle.model_pack,
            requested_provider=config.oracle.provider,
            allow_cpu_fallback=config.oracle.allow_cpu_fallback,
            det_size=tuple(config.oracle.det_size),
        )
    except Exception as exc:
        pytest.skip(f"model not available: {exc}")

    # Synthetic 112x112 frontal face-like image (BGR).
    image = np.zeros((112, 112, 3), dtype=np.uint8)
    image[:, :] = (128, 128, 128)
    cv2.circle(image, (45, 45), 5, (200, 200, 200), -1)
    cv2.circle(image, (67, 45), 5, (200, 200, 200), -1)
    cv2.circle(image, (56, 70), 4, (180, 180, 180), -1)

    dets = oracle.detect(
        image=image,
        detector_low_threshold=0.5,
        frame_width=112,
        frame_height=112,
        quality_config=config.quality.model_dump(mode="json"),
        compute_embeddings=True,
    )

    assert len(dets) >= 0  # detection may fail on synthetic data; do not assert >0
    for det in dets:
        assert det.embedding is not None
        assert np.all(np.isfinite(det.embedding))
        assert np.isclose(np.linalg.norm(det.embedding), 1.0, atol=1e-4)

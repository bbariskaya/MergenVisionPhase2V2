"""Unit tests for the reference extraction pipeline using a fake oracle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mergenvision_video_lab.config import LabConfig, config_sha256
from mergenvision_video_lab.contracts import (
    FrameRecord,
    Landmarks5,
    OnnxModelContract,
    QualityMetrics,
)
from mergenvision_video_lab.extraction import extract_video
from mergenvision_video_lab.oracle.insightface_oracle import OracleDetection


@dataclass
class _FakeOracle:
    """Deterministic fake oracle that emits one face per frame."""

    n_embeddings: int = 1

    @property
    def detector_contract(self) -> OnnxModelContract:
        return OnnxModelContract(
            basename="det.onnx", sha256="a" * 64, size_bytes=1, inputs=[], outputs=[]
        )

    @property
    def recognizer_contract(self) -> OnnxModelContract:
        return OnnxModelContract(
            basename="rec.onnx", sha256="b" * 64, size_bytes=1, inputs=[], outputs=[]
        )

    _available_providers = ["CPUExecutionProvider"]
    _actual_providers = ["CPUExecutionProvider"]

    def detect(
        self,
        image: np.ndarray,
        detector_low_threshold: float,
        frame_width: int,
        frame_height: int,
        quality_config: dict[str, Any],
        compute_embeddings: bool = True,
    ) -> list[OracleDetection]:
        # One face near the center, sized to pass default quality gates.
        x1, y1 = frame_width * 0.25, frame_height * 0.25
        x2, y2 = frame_width * 0.75, frame_height * 0.75
        landmarks = np.array(
            [
                [x1 + (x2 - x1) * 0.3, y1 + (y2 - y1) * 0.3],
                [x1 + (x2 - x1) * 0.7, y1 + (y2 - y1) * 0.3],
                [x1 + (x2 - x1) * 0.5, y1 + (y2 - y1) * 0.5],
                [x1 + (x2 - x1) * 0.3, y1 + (y2 - y1) * 0.7],
                [x1 + (x2 - x1) * 0.7, y1 + (y2 - y1) * 0.7],
            ],
            dtype=np.float32,
        )
        rng = np.random.default_rng(42)
        aligned = rng.integers(0, 256, size=(112, 112, 3), dtype=np.uint8)
        embeddings = []
        for _ in range(self.n_embeddings):
            emb = rng.random(512).astype(np.float32)
            emb /= float(np.linalg.norm(emb))
            embeddings.append(emb)
        return [
            OracleDetection(
                bbox_xyxy=(x1, y1, x2, y2),
                detector_score=0.95,
                landmarks_5=Landmarks5(
                    left_eye=tuple(landmarks[0]),
                    right_eye=tuple(landmarks[1]),
                    nose=tuple(landmarks[2]),
                    left_mouth=tuple(landmarks[3]),
                    right_mouth=tuple(landmarks[4]),
                ),
                aligned_crop=aligned,
                embedding=embeddings[i] if compute_embeddings and i < len(embeddings) else None,
                quality=QualityMetrics(
                    bbox_width_px=x2 - x1,
                    bbox_height_px=y2 - y1,
                    bbox_min_side_px=min(x2 - x1, y2 - y1),
                    bbox_area_px=(x2 - x1) * (y2 - y1),
                    detector_score=0.95,
                    grayscale_laplacian_variance=150.0,
                    brightness_mean=128.0,
                    brightness_std=30.0,
                    dark_clip_fraction=0.0,
                    bright_clip_fraction=0.0,
                    interocular_distance_px=float(np.linalg.norm(landmarks[0] - landmarks[1])),
                    alignment_reprojection_error_px=0.5,
                    alignment_error_normalized_by_interocular=0.02,
                    landmark_geometry_valid=True,
                    finite_embedding=True,
                    composite_quality_score=0.9,
                ).model_dump(mode="json"),
            )
            for i in range(self.n_embeddings)
        ]


def _minimal_config(video_path: Path, sampling_mode: str = "every_frame") -> LabConfig:
    return LabConfig(
        video={
            "path": str(video_path),
            "sampling_mode": sampling_mode,
            "max_frames": None,
            "every_n_frames": None,
            "frames_per_second": None,
            "scene_cut_threshold": 1.0,  # disable scene cuts
            "scene_cut_downscale": 64,
        },
        oracle={
            "provider": "cpu",
            "allow_cpu_fallback": False,
        },
        quality={
            "min_face_side_px": 10,
            "min_detector_score_recognition": 0.5,
            "min_laplacian_variance": 20.0,
            "max_alignment_error_normalized": 0.5,
            "min_brightness_mean": 10.0,
            "max_brightness_mean": 250.0,
            "max_dark_clip_fraction": 1.0,
            "max_bright_clip_fraction": 1.0,
        },
    )


def test_extraction_writes_frame_ledger(tmp_path: Path, cfr_video: Path) -> None:
    config = _minimal_config(cfr_video)
    oracle = _FakeOracle()
    store, manifest = extract_video(config, oracle=oracle)

    frames = store.read_frames()
    assert len(frames) == 10
    assert manifest.decoded_frame_count == 10
    assert manifest.sampled_frame_count == 10
    assert manifest.processed_frame_count == 10
    assert [f.frame_index for f in frames] == list(range(10))
    assert all(isinstance(f, FrameRecord) for f in frames)


def test_extraction_observations_and_embeddings_match(cfr_video: Path) -> None:
    config = _minimal_config(cfr_video)
    oracle = _FakeOracle()
    store, manifest = extract_video(config, oracle=oracle)

    observations = store.read_observations()
    assert len(observations) == 10
    assert manifest.observation_count == 10
    assert manifest.valid_embedding_count == 10

    embeddings = store.read_embeddings()
    assert embeddings.shape == (10, 512)
    for obs in observations:
        assert obs.recognition_eligible
        assert obs.embedding_index is not None
        assert 0 <= obs.embedding_index < embeddings.shape[0]


def test_extraction_manifest_counters_are_consistent(cfr_video: Path) -> None:
    config = _minimal_config(cfr_video)
    oracle = _FakeOracle()
    store, manifest = extract_video(config, oracle=oracle)

    assert manifest.video_sha256 is not None
    assert manifest.config_sha256 == config_sha256(config)
    assert manifest.extraction_timing.total_seconds > 0.0
    assert manifest.extraction_timing.serialization_seconds > 0.0
    store.validate()


def test_extraction_checksums_written_last(cfr_video: Path) -> None:
    config = _minimal_config(cfr_video)
    oracle = _FakeOracle()
    store, manifest = extract_video(config, oracle=oracle)

    checksums = store.checksums_path.read_text(encoding="utf-8")
    assert "manifest.json" in checksums
    assert "frames.jsonl" in checksums
    assert "observations.jsonl" in checksums
    assert "embeddings.npy" in checksums


def test_extraction_resume_without_inference(cfr_video: Path) -> None:
    config = _minimal_config(cfr_video)
    oracle = _FakeOracle(n_embeddings=1)
    store1, manifest1 = extract_video(config, oracle=oracle)

    # Second call with a different oracle should return the frozen result.
    bad_oracle = _FakeOracle(n_embeddings=99)
    store2, manifest2 = extract_video(config, oracle=bad_oracle)

    assert store1.run_dir == store2.run_dir
    assert manifest1.observation_count == manifest2.observation_count
    assert manifest1.valid_embedding_count == manifest2.valid_embedding_count
    assert (
        store2.read_observations()[0].observation_id == store1.read_observations()[0].observation_id
    )


def test_extraction_force_re_runs(cfr_video: Path) -> None:
    config = _minimal_config(cfr_video)
    oracle1 = _FakeOracle(n_embeddings=1)
    extract_video(config, oracle=oracle1)

    oracle2 = _FakeOracle(n_embeddings=2)
    _, manifest = extract_video(config, force=True, oracle=oracle2)

    assert manifest.observation_count == 20
    assert manifest.valid_embedding_count == 20

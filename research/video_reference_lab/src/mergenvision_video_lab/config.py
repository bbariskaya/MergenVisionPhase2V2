"""YAML configuration loading and validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from mergenvision_video_lab.contracts import CONFIG_SCHEMA_VERSION
from mergenvision_video_lab.errors import ConfigError


class VideoConfig(BaseModel):
    path: str
    sampling_mode: Literal["every_frame", "every_n_frames", "frames_per_second"]
    max_frames: int | None = None
    every_n_frames: int | None = None
    frames_per_second: float | None = None
    scene_cut_threshold: float = 0.45
    scene_cut_downscale: int = 64


class OracleConfig(BaseModel):
    model_root: str | None = None
    model_pack: str | None = None
    provider: Literal["cuda", "cpu"]
    allow_cpu_fallback: bool = False
    allow_model_download: bool = False
    det_size: list[int] = Field(default_factory=lambda: [640, 640])
    detector_low_threshold: float = 0.10


class AlignmentConfig(BaseModel):
    output_size: int = 112
    color_order: Literal["RGB", "BGR"] = "BGR"
    border_mode: str = "constant_zero"
    interpolation: str = "bilinear"
    landmark_order: list[str] = Field(
        default_factory=lambda: [
            "left_eye",
            "right_eye",
            "nose",
            "left_mouth",
            "right_mouth",
        ]
    )


class QualityConfig(BaseModel):
    min_face_side_px: int = 32
    min_detector_score_recognition: float = 0.60
    min_laplacian_variance: float = 40.0
    max_alignment_error_normalized: float = 0.10
    min_brightness_mean: float = 30.0
    max_brightness_mean: float = 225.0
    max_dark_clip_fraction: float = 0.40
    max_bright_clip_fraction: float = 0.40


class TrackingConfig(BaseModel):
    high_detection_threshold: float = 0.60
    low_detection_threshold: float = 0.10
    new_track_threshold: float = 0.70
    first_stage_min_iou: float = 0.10
    second_stage_min_iou: float = 0.30
    unconfirmed_min_iou: float = 0.30
    short_term_min_cosine: float = 0.10
    appearance_weight: float = 0.35
    max_lost_frames: int = 30
    max_lost_ns: int = 1_000_000_000
    scene_cut_reset: bool = True
    evidence_top_k: int = 5
    evidence_min_separation_ns: int = 200_000_000


class TemplatesConfig(BaseModel):
    max_selected_samples: int = 5
    min_selected_samples: int = 1
    min_temporal_separation_ns: int = 200_000_000
    outlier_mad_scale: float = 3.0
    outlier_absolute_cosine_floor: float = 0.20
    min_quality_score: float = 0.0


class ReconciliationConfig(BaseModel):
    min_tracklet_cosine: float = 0.45
    min_cluster_member_cosine: float = 0.40
    min_top1_top2_margin: float = 0.05
    overlap_tolerance_ns: int = 0


class GalleryConfig(BaseModel):
    root: str = "test_gallery"
    min_identity_count_for_strict_margin: int = 2
    min_samples_per_identity: int = 2
    match_threshold: float = 0.45
    match_margin: float = 0.05


class AppearancesConfig(BaseModel):
    max_gap_multiplier: float = 2.5


class ReplayConfig(BaseModel):
    chunk_sizes: list[int] = Field(default_factory=lambda: [1, 8, 17, 64])
    repetitions: int = 2


class BenchmarkConfig(BaseModel):
    warmup_runs: int = 5
    measured_runs: int = 20
    informational_target_fps: int = 600


class RenderConfig(BaseModel):
    debug_mp4: bool = True
    preserve_audio_if_possible: bool = True
    contact_sheet_columns: int = 4


class OutputConfig(BaseModel):
    base_dir: str = "artifacts/video_reference"


class LabConfig(BaseModel):
    schema_version: Literal["mv-video-reference-config/v1"] = CONFIG_SCHEMA_VERSION
    video: VideoConfig
    oracle: OracleConfig
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    templates: TemplatesConfig = Field(default_factory=TemplatesConfig)
    reconciliation: ReconciliationConfig = Field(default_factory=ReconciliationConfig)
    gallery: GalleryConfig = Field(default_factory=GalleryConfig)
    appearances: AppearancesConfig = Field(default_factory=AppearancesConfig)
    replay: ReplayConfig = Field(default_factory=ReplayConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("schema_version")
    @classmethod
    def _schema_version(cls, v: str) -> str:
        if v != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"config schema_version must be {CONFIG_SCHEMA_VERSION!r}")
        return v

    def model_dump_without_defaults(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def config_sha256(config: LabConfig) -> str:
    """Return a deterministic SHA-256 of the validated config."""
    canonical = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _repo_root() -> Path:
    """Locate the repository root by marker files.

    The repository root is the nearest ancestor of this file that contains a
    `.git` directory, or that contains both `Makefile` and `pyproject.toml`.
    This avoids mis-identifying a nested package directory (which may contain
    its own `pyproject.toml`) as the repository root.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists():
            return parent
        if (parent / "Makefile").exists() and (parent / "pyproject.toml").exists():
            return parent
    # Fallback to the original relative depth if no marker is found.
    return here.parents[4]


def resolve_repo_relative_path(path: Path | str) -> Path:
    """Resolve a path relative to the repository root if not absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    return _repo_root() / p


def _preprocess_alignment_fingerprint(config: LabConfig) -> str:
    """Deterministic fingerprint of the preprocess/alignment contract."""
    payload = json.dumps(
        {
            "output_size": config.alignment.output_size,
            "color_order": config.alignment.color_order,
            "border_mode": config.alignment.border_mode,
            "interpolation": config.alignment.interpolation,
            "landmark_order": config.alignment.landmark_order,
            "det_size": config.oracle.det_size,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def resolve_run_dir(config: LabConfig) -> Path:
    """Return the artifact run directory for a given config."""
    from mergenvision_video_lab.errors import VideoReadError
    from mergenvision_video_lab.hashing import sha256_file

    video_path = resolve_repo_relative_path(config.video.path)
    if not video_path.exists():
        raise VideoReadError(f"input video not found: {video_path}")
    video_sha256 = sha256_file(video_path)
    cfg_sha = config_sha256(config)
    preprocess_fp = _preprocess_alignment_fingerprint(config)
    base = Path(config.output.base_dir)
    return resolve_repo_relative_path(base / video_sha256[:12] / cfg_sha[:12] / preprocess_fp)


def load_config(path: Path | str) -> LabConfig:
    """Load and validate a YAML lab configuration."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError("config file must contain a YAML mapping")
    try:
        return LabConfig(**raw)
    except Exception as exc:
        raise ConfigError(f"config validation failed: {exc}") from exc

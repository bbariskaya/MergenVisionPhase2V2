"""Pydantic v2 data contracts for observations, tracks, and reports."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

# ------------------------------------------------------------------------------
# Shared constants
# ------------------------------------------------------------------------------

MANIFEST_SCHEMA_VERSION: Literal["mv-video-reference-manifest/v1"] = (
    "mv-video-reference-manifest/v1"
)
OBSERVATION_SCHEMA_VERSION: Literal["mv-face-observation/v1"] = "mv-face-observation/v1"
FRAME_SCHEMA_VERSION: Literal["mv-video-frame/v1"] = "mv-video-frame/v1"
GROUND_TRUTH_SCHEMA_VERSION: Literal["mv-video-ground-truth/v1"] = "mv-video-ground-truth/v1"
CONFIG_SCHEMA_VERSION: Literal["mv-video-reference-config/v1"] = "mv-video-reference-config/v1"

LANDMARK_ORDER = ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"]


# ------------------------------------------------------------------------------
# Value-object helpers
# ------------------------------------------------------------------------------


class BBoxXYXY(BaseModel):
    """Original display-space bounding box, half-open [x1,y1,x2,y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @field_validator("x1", "y1", "x2", "y2")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError("bbox coordinates must be finite")
        return float(v)

    @model_validator(mode="after")
    def _ordered(self) -> BBoxXYXY:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bbox must have positive width and height")
        return self

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


class Landmarks5(BaseModel):
    """Five facial landmarks in detector contract order.

    ``left`` means subject-left (mirror side of the image for a front-facing
    subject). Verify with contact-sheet overlay; do not assume image-left.
    """

    left_eye: tuple[float, float]
    right_eye: tuple[float, float]
    nose: tuple[float, float]
    left_mouth: tuple[float, float]
    right_mouth: tuple[float, float]

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.left_eye,
                self.right_eye,
                self.nose,
                self.left_mouth,
                self.right_mouth,
            ],
            dtype=np.float32,
        )

    def to_list(self) -> list[list[float]]:
        return [
            list(self.left_eye),
            list(self.right_eye),
            list(self.nose),
            list(self.left_mouth),
            list(self.right_mouth),
        ]


class QualityMetrics(BaseModel):
    """Raw and composite quality metrics for one face observation."""

    bbox_width_px: float
    bbox_height_px: float
    bbox_min_side_px: float
    bbox_area_px: float
    detector_score: float
    grayscale_laplacian_variance: float
    brightness_mean: float
    brightness_std: float
    dark_clip_fraction: float
    bright_clip_fraction: float
    interocular_distance_px: float
    alignment_reprojection_error_px: float
    alignment_error_normalized_by_interocular: float
    landmark_geometry_valid: bool
    finite_embedding: bool
    raw_embedding_norm_before: float | None = None
    composite_quality_score: float
    hard_rejection_reasons: list[str] = Field(default_factory=list)
    pose: dict[str, float] | None = None
    occlusion: dict[str, Any] | None = None


# ------------------------------------------------------------------------------
# Frame ledger
# ------------------------------------------------------------------------------


class FrameRecord(BaseModel):
    """One decoded/processed video frame, including zero-face frames."""

    schema_version: Literal["mv-video-frame/v1"] = FRAME_SCHEMA_VERSION
    source_id: str
    frame_index: int = Field(..., ge=0)
    pts: int
    time_base_num: int
    time_base_den: int = Field(..., gt=0)
    pts_ns: int
    coded_width: int = Field(..., gt=0)
    coded_height: int = Field(..., gt=0)
    display_width: int = Field(..., gt=0)
    display_height: int = Field(..., gt=0)
    rotation_applied: float = 0.0
    sampled: bool = True
    processed: bool = True
    scene_change_score: float = 0.0
    scene_cut_before: bool = False


# ------------------------------------------------------------------------------
# Observations
# ------------------------------------------------------------------------------


class FaceObservation(BaseModel):
    """One detected face in one processed video frame."""

    schema_version: Literal["mv-face-observation/v1"] = OBSERVATION_SCHEMA_VERSION
    observation_id: str
    source_id: str
    frame_index: int = Field(..., ge=0)
    pts: int
    time_base_num: int
    time_base_den: int = Field(..., gt=0)
    pts_ns: int
    frame_width: int = Field(..., gt=0)
    frame_height: int = Field(..., gt=0)
    rotation_applied: float = 0.0
    detection_ordinal: int = Field(..., ge=0)
    bbox_xyxy: BBoxXYXY
    detector_score: float
    landmarks_5: Landmarks5
    landmark_order: list[str] = Field(default_factory=lambda: LANDMARK_ORDER.copy())
    quality: QualityMetrics
    tracking_eligible: bool
    recognition_eligible: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    embedding_index: int | None = None
    scene_cut_before: bool = False
    provenance: Literal["reference_oracle"] = "reference_oracle"

    @field_validator("landmark_order")
    @classmethod
    def _landmark_order(cls, v: list[str]) -> list[str]:
        if v != LANDMARK_ORDER:
            raise ValueError(f"landmark_order must be exactly {LANDMARK_ORDER}")
        return v

    @field_validator("detector_score")
    @classmethod
    def _detector_score_finite(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError("detector_score must be finite")
        return float(v)


# ------------------------------------------------------------------------------
# Model / alignment contracts stored in manifest
# ------------------------------------------------------------------------------


class OnnxModelContract(BaseModel):
    """ONNX graph contract for a detector or recognizer artifact."""

    basename: str
    sha256: str
    size_bytes: int
    inputs: list[dict[str, Any]]
    outputs: list[dict[str, Any]]
    opset: int | None = None
    producer: str | None = None


class AlignmentContract(BaseModel):
    """Alignment parameters used for this run."""

    output_size: int
    color_order: Literal["RGB", "BGR"]
    border_mode: str
    interpolation: str
    landmark_order: list[str]
    arcface_template: list[list[float]]


class SamplingContract(BaseModel):
    """Video sampling parameters."""

    mode: Literal["every_frame", "every_n_frames", "frames_per_second"]
    every_n_frames: int | None = None
    frames_per_second: float | None = None
    max_frames: int | None = None


# ------------------------------------------------------------------------------
# Run manifest
# ------------------------------------------------------------------------------


class ExtractionTiming(BaseModel):
    """Per-stage wall-clock timing for extraction."""

    decode_seconds: float
    oracle_seconds: float
    quality_alignment_seconds: float
    serialization_seconds: float
    total_seconds: float


class RunManifest(BaseModel):
    """Top-level manifest for a frozen reference run."""

    schema_version: Literal["mv-video-reference-manifest/v1"] = MANIFEST_SCHEMA_VERSION
    run_id: str
    video_sha256: str
    video_size_bytes: int
    logical_video_name: str
    container: str
    codec: str
    pixel_format: str
    display_width: int
    display_height: int
    rotation_degrees: float
    stream_index: int
    time_base_num: int
    time_base_den: int
    duration_ns: int
    decoded_frame_count: int
    sampled_frame_count: int
    processed_frame_count: int
    sampling_contract: SamplingContract
    detector_contract: OnnxModelContract
    recognizer_contract: OnnxModelContract
    alignment_contract: AlignmentContract
    provider_requested: str
    providers_available: list[str]
    providers_actual: list[str]
    package_versions: dict[str, str]
    config_sha256: str
    observation_count: int
    valid_embedding_count: int
    rejection_counts: dict[str, int]
    extraction_timing: ExtractionTiming
    scene_cut_frame_indices: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------------------
# Tracker outputs
# ------------------------------------------------------------------------------


class TrackAssignment(BaseModel):
    """One observation-to-track assignment emitted by a tracker update."""

    observation_id: str
    frame_index: int
    pts_ns: int
    raw_tracklet_id: str
    strategy: str
    cost: float | None = None


class RawTrackletSummary(BaseModel):
    """Summary of one raw tracklet."""

    raw_tracklet_id: str
    strategy: str
    first_frame_index: int
    last_frame_index: int
    first_pts_ns: int
    last_pts_ns: int
    observation_count: int
    detection_ordinal_ids: list[str]
    state: Literal["active", "lost", "removed"]


# ------------------------------------------------------------------------------
# Reconciliation outputs
# ------------------------------------------------------------------------------


class CanonicalTrack(BaseModel):
    """Offline reconciliation group of raw tracklets."""

    canonical_track_id: str
    raw_tracklet_ids: list[str]
    display_label: str | None = None
    first_seen_pts_ns: int
    last_seen_pts_ns: int
    total_duration_ns: int
    appearances: list[dict[str, Any]]
    detections: list[dict[str, Any]]
    template_evidence: dict[str, Any]
    gallery_top1_label: str | None = None
    gallery_top1_cosine: float | None = None
    gallery_top2_label: str | None = None
    gallery_top2_cosine: float | None = None
    gallery_margin: float | None = None
    decision_reason: str
    confidence_evidence: dict[str, Any]
    limitations: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------------------
# Ground truth
# ------------------------------------------------------------------------------


class GroundTruthAnchor(BaseModel):
    """Human-labeled anchor tied to an observation or frame."""

    anchor_id: str
    label: str
    split: Literal["calibration", "holdout"]
    frame_index: int | None = None
    observation_id: str | None = None


class GroundTruth(BaseModel):
    """Minimal human-label checkpoint."""

    schema_version: Literal["mv-video-ground-truth/v1"] = GROUND_TRUTH_SCHEMA_VERSION
    video_sha256: str
    anchors: list[GroundTruthAnchor] = Field(default_factory=list)
    cannot_link_pairs: list[tuple[str, str]] = Field(default_factory=list)
    ignored_observation_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

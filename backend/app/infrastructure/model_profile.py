"""Parsed model/preprocess profile used by the native runtime and readiness checks."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class _DynamicProfile(BaseModel):
    min: list[int]
    opt: list[int]
    max: list[int]


class _DetectorOutputTensors(BaseModel):
    location: str
    confidence: str
    landmarks: str


class _DetectorProfile(BaseModel):
    name: str
    input_layout: str
    color_order: str
    input_tensor_name: str
    input_dtype: str
    input_shape: list[int]
    dynamic_profile: _DynamicProfile
    output_tensors: _DetectorOutputTensors
    confidence_threshold: float
    nms_threshold: float
    max_candidates: int
    anchor_strides: list[int]
    anchor_sizes: list[list[int]]
    anchor_ratios: list[float]
    variances: list[float]
    landmark_order: list[str]


class _RecognizerOutputTensors(BaseModel):
    embedding: str


class _RecognizerProfile(BaseModel):
    name: str
    input_layout: str
    color_order: str
    input_tensor_name: str
    input_dtype: str
    input_shape: list[int]
    dynamic_profile: _DynamicProfile
    output_tensors: _RecognizerOutputTensors
    embedding_dim: int
    normalize: str


class _AlignmentProfile(BaseModel):
    template: list[list[float]]
    crop_size: list[int]


class _ModelArtifact(BaseModel):
    onnx_path: str
    onnx_sha256: str


class _EngineEntry(BaseModel):
    engine_path: str
    engine_sha256: str
    profile: _DynamicProfile
    precision: str


class _EngineManifest(BaseModel):
    retinaface_r50_dynamic: _EngineEntry = Field(alias="retinaface_r50_dynamic")
    glintr100: _EngineEntry
    build_command: str
    tensorrt_version: str
    cuda_version: str
    container_digest: str
    gpu_compute_capability: str
    gpu_uuid: str
    build_timestamp: str

    model_config = {"populate_by_name": True}


class _Models(BaseModel):
    retinaface_r50_dynamic: _ModelArtifact = Field(alias="retinaface_r50_dynamic")
    glintr100: _ModelArtifact

    model_config = {"populate_by_name": True}


class ModelProfile(BaseModel):
    """Validated view of the model/preprocess/engine profile JSON."""

    schema_version: str
    model_version: str
    preprocess_version: str
    detector: _DetectorProfile
    recognizer: _RecognizerProfile
    alignment: _AlignmentProfile
    models: _Models
    engine_manifest: _EngineManifest

    model_config = {"populate_by_name": True, "protected_namespaces": ()}

    @classmethod
    def load(cls, path: str) -> ModelProfile:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"model profile not found: {path}")
        data = json.loads(resolved.read_text(encoding="utf-8"))
        return cls.model_validate(data)

    @property
    def detector_profile(self) -> _DynamicProfile:
        return self.detector.dynamic_profile

    @property
    def recognizer_profile(self) -> _DynamicProfile:
        return self.recognizer.dynamic_profile

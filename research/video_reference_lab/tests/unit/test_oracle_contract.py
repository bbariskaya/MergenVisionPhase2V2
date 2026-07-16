"""Tests for oracle construction, provider selection, and preprocess contract."""

from __future__ import annotations

import pytest

from mergenvision_video_lab.errors import ModelArtifactError
from mergenvision_video_lab.model_inventory import select_providers
from mergenvision_video_lab.oracle.insightface_oracle import FaceAnalysisOracle


def test_select_providers_cpu_only() -> None:
    """CPU request returns CPUExecutionProvider."""
    providers = select_providers("cpu", ["CPUExecutionProvider"], allow_cpu_fallback=False)
    assert providers == ["CPUExecutionProvider"]


def test_select_providers_cuda_unavailable_no_fallback_fails() -> None:
    """CUDA request without fallback fails when CUDA is unavailable."""
    with pytest.raises(ModelArtifactError):
        select_providers(
            "cuda",
            ["CPUExecutionProvider"],
            allow_cpu_fallback=False,
        )


def test_select_providers_cuda_fallback_when_allowed() -> None:
    """CUDA request with allowed fallback returns CPU when CUDA is missing."""
    providers = select_providers(
        "cuda",
        ["CPUExecutionProvider"],
        allow_cpu_fallback=True,
    )
    assert providers == ["CPUExecutionProvider"]


def test_oracle_fails_when_pack_missing() -> None:
    """Instantiation fails closed if the model pack is not cached."""
    with pytest.raises(ModelArtifactError):
        FaceAnalysisOracle(
            model_root="/nonexistent/model/cache",
            model_pack="buffalo_l",
            requested_provider="cpu",
            allow_cpu_fallback=False,
            det_size=(640, 640),
        )


def test_oracle_rejects_unsupported_pack() -> None:
    """Only buffalo_l is supported in the reference lab."""
    with pytest.raises(ModelArtifactError):
        FaceAnalysisOracle(
            model_root="/nonexistent/model/cache",
            model_pack="buffalo_m",
            requested_provider="cpu",
            allow_cpu_fallback=False,
            det_size=(640, 640),
        )

"""Tests for explicit model acquisition and manifest building."""

from __future__ import annotations

import pytest

from mergenvision_video_lab.errors import ModelArtifactError
from mergenvision_video_lab.model_manager import acquire_model_pack


def test_acquire_rejects_unauthorized_pack() -> None:
    """Only authorized official packs may be acquired."""
    with pytest.raises(ModelArtifactError):
        acquire_model_pack(name="not_buffalo_l", provider="cpu")

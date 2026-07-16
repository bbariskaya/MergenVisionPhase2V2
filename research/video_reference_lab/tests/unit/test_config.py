"""Unit tests for configuration loading."""

from __future__ import annotations

import pytest
import yaml

from mergenvision_video_lab.config import load_config
from mergenvision_video_lab.errors import ConfigError


def test_load_friends_baseline() -> None:
    cfg = load_config("configs/friends_baseline.yaml")
    assert cfg.schema_version == "mv-video-reference-config/v1"
    assert cfg.video.path == "test_videos/Friends.mp4"
    assert cfg.oracle.det_size == [640, 640]
    assert cfg.quality.min_face_side_px == 32
    assert cfg.replay.chunk_sizes == [1, 8, 17, 64]


def test_load_config_missing_file() -> None:
    with pytest.raises(ConfigError):
        load_config("configs/does_not_exist.yaml")


def test_load_config_invalid_schema(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"schema_version": "wrong"}), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)

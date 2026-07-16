"""Tests for repo-root-relative path resolution and config loading."""

from __future__ import annotations

import os
from pathlib import Path

from mergenvision_video_lab.config import _repo_root, load_config, resolve_repo_relative_path

REPO_ROOT = _repo_root()
LAB_DIR = REPO_ROOT / "research" / "video_reference_lab"


def test_repo_root_contains_markers() -> None:
    """The discovered repo root must contain expected marker files."""
    assert (REPO_ROOT / ".git").exists(), "repo root must contain .git directory"
    assert (REPO_ROOT / "Makefile").exists(), "repo root must contain Makefile"


def test_resolve_absolute_path_unchanged() -> None:
    """Absolute paths are returned unchanged."""
    absolute = Path("/tmp/some_video.mp4").resolve()
    assert resolve_repo_relative_path(absolute) == absolute


def test_resolve_relative_from_repo_root() -> None:
    """Relative paths resolve from repo root regardless of CWD."""
    rel = "test_videos/Friends.mp4"
    expected = REPO_ROOT / rel
    assert resolve_repo_relative_path(rel) == expected


def test_load_config_from_repo_root() -> None:
    """Config can be loaded when CWD is the repository root."""
    cfg_path = "research/video_reference_lab/configs/friends_baseline.yaml"
    cfg = load_config(REPO_ROOT / cfg_path)
    assert cfg.schema_version == "mv-video-reference-config/v1"
    assert cfg.oracle.provider in ("cpu", "cuda")


def test_load_config_from_lab_directory() -> None:
    """Config can be loaded when CWD is the lab directory."""
    cfg_path = "configs/friends_baseline.yaml"
    cfg = load_config(LAB_DIR / cfg_path)
    assert cfg.schema_version == "mv-video-reference-config/v1"
    assert cfg.oracle.provider in ("cpu", "cuda")


def test_load_config_from_unrelated_directory() -> None:
    """Config paths in the file are still resolved relative to repo root."""
    original_cwd = os.getcwd()
    tmp = os.environ.get("TMPDIR", "/tmp")
    try:
        os.chdir(tmp)
        cfg = load_config(LAB_DIR / "configs" / "friends_baseline.yaml")
        video_path = resolve_repo_relative_path(cfg.video.path)
        assert video_path == REPO_ROOT / cfg.video.path
    finally:
        os.chdir(original_cwd)


def test_cpu_and_cuda_configs_are_explicit() -> None:
    """CPU and CUDA configs request their respective providers."""
    cpu_cfg = load_config(LAB_DIR / "configs" / "friends_baseline_cpu.yaml")
    cuda_cfg = load_config(LAB_DIR / "configs" / "friends_baseline_cuda.yaml")
    assert cpu_cfg.oracle.provider == "cpu"
    assert cpu_cfg.oracle.allow_cpu_fallback is False
    assert cuda_cfg.oracle.provider == "cuda"
    assert cuda_cfg.oracle.allow_cpu_fallback is False

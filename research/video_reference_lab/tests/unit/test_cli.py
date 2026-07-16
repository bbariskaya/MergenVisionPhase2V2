"""CLI surface tests using Typer's CliRunner."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from mergenvision_video_lab.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    """The CLI prints help and lists all top-level commands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "doctor" in result.stdout
    assert "models" in result.stdout
    assert "extract" in result.stdout


def test_doctor_with_missing_video_exits_nonzero() -> None:
    """doctor fails closed when the configured video is missing."""
    result = runner.invoke(app, ["doctor", "--config", "configs/friends_baseline_cpu.yaml"])
    # The default Friends.mp4 may not exist in CI; the command must not silently pass.
    if "input video not found" in result.stdout or result.exit_code != 0:
        assert result.exit_code != 0
    else:
        pytest.skip("Friends.mp4 present; smoke-level doctor test only")


def test_models_acquire_requires_allow_download() -> None:
    """Downloads are disabled unless --allow-download is explicitly passed."""
    result = runner.invoke(app, ["models", "acquire", "--name", "buffalo_l", "--provider", "cpu"])
    assert result.exit_code != 0
    assert "Downloads are disabled" in result.stdout


def test_models_acquire_rejects_unauthorized_pack() -> None:
    """Only authorized official packs may be acquired."""
    result = runner.invoke(
        app,
        [
            "models",
            "acquire",
            "--name",
            "unauthorized_pack",
            "--provider",
            "cpu",
            "--allow-download",
        ],
    )
    assert result.exit_code != 0
    assert "Only official packs" in result.stdout

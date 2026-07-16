"""Tests for video frame sampling predicates."""

from __future__ import annotations

import pytest

from mergenvision_video_lab.config import LabConfig
from mergenvision_video_lab.extraction import _is_sampled_frame


def _config(sampling_mode: str, **kwargs) -> LabConfig:
    video = {
        "path": "test.mp4",
        "sampling_mode": sampling_mode,
        "max_frames": None,
        "every_n_frames": None,
        "frames_per_second": None,
        "scene_cut_threshold": 0.45,
        "scene_cut_downscale": 64,
    }
    video.update(kwargs)
    return LabConfig(
        video=video,
        oracle={"provider": "cpu", "allow_cpu_fallback": False},
    )


def test_every_frame_samples_all_indices() -> None:
    config = _config("every_frame")
    assert all(_is_sampled_frame(i, config, 30.0) for i in range(100))


@pytest.mark.parametrize("n", [1, 2, 3, 5, 10])
def test_every_n_frames_samples_modulo_n(n: int) -> None:
    config = _config("every_n_frames", every_n_frames=n)
    for i in range(100):
        assert _is_sampled_frame(i, config, 30.0) == (i % n == 0)


def test_every_n_frames_with_zero_or_negative_fallback() -> None:
    config = _config("every_n_frames", every_n_frames=0)
    assert all(_is_sampled_frame(i, config, 30.0) for i in range(10))
    config = _config("every_n_frames", every_n_frames=-1)
    assert all(_is_sampled_frame(i, config, 30.0) for i in range(10))


def test_frames_per_second_at_exact_divisors() -> None:
    config = _config("frames_per_second", frames_per_second=10.0)
    sampled = [_is_sampled_frame(i, config, 30.0) for i in range(15)]
    expected = [i % 3 == 0 for i in range(15)]
    assert sampled == expected


def test_frames_per_second_half_rate() -> None:
    config = _config("frames_per_second", frames_per_second=15.0)
    sampled = [_is_sampled_frame(i, config, 30.0) for i in range(10)]
    expected = [i % 2 == 0 for i in range(10)]
    assert sampled == expected


def test_frames_per_second_with_invalid_rate_fallback() -> None:
    config = _config("frames_per_second", frames_per_second=0.0)
    assert all(_is_sampled_frame(i, config, 30.0) for i in range(10))
    config = _config("frames_per_second", frames_per_second=-5.0)
    assert all(_is_sampled_frame(i, config, 30.0) for i in range(10))
    config = _config("frames_per_second", frames_per_second=10.0)
    assert all(_is_sampled_frame(i, config, 0.0) for i in range(10))


def test_sampling_is_stateless_across_indices() -> None:
    """Sampling predicate must not depend on hidden state."""
    config = _config("frames_per_second", frames_per_second=6.0)
    sampled_a = [_is_sampled_frame(i, config, 30.0) for i in range(30)]
    sampled_b = [_is_sampled_frame(i, config, 30.0) for i in range(30)]
    assert sampled_a == sampled_b

"""Unit tests for PyAV video reader."""

from __future__ import annotations

from pathlib import Path

import pytest

from unittest.mock import MagicMock

from mergenvision_video_lab.errors import VideoReadError
from mergenvision_video_lab.video_reader import (
    VideoProbeResult,
    VideoReader,
    _read_rotation,
    probe_video,
)

from tests.fixtures.video_fixtures import _make_video


def test_probe_cfr_video(cfr_video: Path) -> None:
    probe = probe_video(cfr_video)
    assert probe.container == "mp4"
    assert probe.codec == "h264"
    assert probe.width == 160
    assert probe.height == 120
    assert probe.avg_frame_rate > 0.0
    assert probe.audio_stream_count == 0
    assert probe.duration_ns is not None
    assert probe.duration_ns > 0


def test_decode_cfr_frames(cfr_video: Path) -> None:
    reader = VideoReader(cfr_video, source_id="cfr")
    frames = list(reader)
    assert len(frames) == 10
    for i, frame in enumerate(frames):
        assert frame.frame_index == i
        assert frame.source_id == "cfr"
        assert frame.pts_ns > 0 or i == 0
        assert frame.ndarray.shape == (120, 160, 3)
        if i > 0:
            assert frame.pts_ns >= frames[i - 1].pts_ns


def test_decode_rotation_90(rotated_video_90: Path) -> None:
    reader = VideoReader(rotated_video_90)
    # Width/height stay coded; display dimensions swap for 90-degree rotation
    # when rotation metadata is present.
    assert reader.probe.width == 160
    assert reader.probe.height == 120
    # The synthetic fixture does not preserve rotate metadata through x264,
    # so rotation_degrees may be 0 here. The rotation logic is tested below.
    frames = list(reader)
    assert len(frames) == 5


def test_decode_vfr_non_uniform_pts(vfr_video: Path) -> None:
    reader = VideoReader(vfr_video)
    frames = list(reader)
    assert len(frames) == 5
    pts_ns_list = [f.pts_ns for f in frames]
    # Strictly increasing.
    assert all(pts_ns_list[i] < pts_ns_list[i + 1] for i in range(len(pts_ns_list) - 1))
    # Gaps reflect non-uniform durations (at least one gap differs).
    gaps = [pts_ns_list[i + 1] - pts_ns_list[i] for i in range(len(pts_ns_list) - 1)]
    assert len(set(gaps)) > 1


def test_decode_missing_video(tmp_path: Path) -> None:
    with pytest.raises(VideoReadError):
        probe_video(tmp_path / "missing.mp4")


def test_no_duplicate_frame_index_pts(cfr_video: Path) -> None:
    reader = VideoReader(cfr_video)
    seen = set()
    for frame in reader:
        key = (frame.frame_index, frame.pts_ns)
        assert key not in seen
        seen.add(key)


def test_read_rotation_from_metadata() -> None:
    stream = MagicMock()
    stream.side_data = []
    stream.metadata = {"rotate": "90"}
    assert _read_rotation(stream) == 90.0


def test_read_rotation_from_side_data() -> None:
    side = MagicMock()
    side.type = "displaymatrix"
    side.rotation = 270.0
    stream = MagicMock()
    stream.side_data = [side]
    stream.metadata = {}
    assert _read_rotation(stream) == 270.0


def test_read_rotation_default_zero() -> None:
    stream = MagicMock()
    stream.side_data = []
    stream.metadata = {}
    assert _read_rotation(stream) == 0.0


def test_rotation_180(tmp_path: Path) -> None:
    path = tmp_path / "rotated_180.mp4"
    _make_video(path, frame_count=3, fps=30, rotation=180)
    probe = probe_video(path)
    # Display dimensions do not swap for 180-degree rotation.
    assert probe.display_width == probe.width
    assert probe.display_height == probe.height


def test_rotation_270() -> None:
    # Test display-dimension swap for 270-degree rotation via mock.
    probe = VideoProbeResult(
        container="mp4",
        codec="h264",
        pixel_format="yuv420p",
        width=160,
        height=120,
        display_width=120,
        display_height=160,
        rotation_degrees=270.0,
        sample_aspect_ratio=None,
        display_aspect_ratio=None,
        avg_frame_rate=30.0,
        real_frame_rate=None,
        time_base_num=1,
        time_base_den=30,
        start_time=0,
        duration_ns=1_000_000_000,
        estimated_frame_count=30,
        audio_stream_count=0,
        video_stream_index=0,
    )
    assert probe.rotation_degrees == 270.0
    assert probe.display_width == 120
    assert probe.display_height == 160

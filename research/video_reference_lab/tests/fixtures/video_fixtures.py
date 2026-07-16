"""Synthetic video fixtures for tests."""

from __future__ import annotations

from pathlib import Path
from typing import Generator

from fractions import Fraction

import av
import numpy as np
import pytest


FRAME_WIDTH = 160
FRAME_HEIGHT = 120
FPS_CFR = 30


def _make_video(
    path: Path,
    frame_count: int,
    fps: float,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
    rotation: int | None = None,
    vfr_durations_ms: list[int] | None = None,
) -> None:
    """Write a small synthetic MP4 using PyAV."""
    container = av.open(str(path), mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.time_base = Fraction(1, 1000)  # millisecond PTS units
    if rotation is not None:
        stream.metadata["rotate"] = str(rotation)

    cumulative_ms = 0
    for i in range(frame_count):
        # Each frame has a distinct color so ordering is verifiable.
        r = (i * 7) % 256
        g = (i * 13) % 256
        b = (i * 19) % 256
        img = np.full((height, width, 3), (b, g, r), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        frame.pts = cumulative_ms
        if vfr_durations_ms is not None:
            cumulative_ms += vfr_durations_ms[i]
        else:
            cumulative_ms += int(round(1000.0 / fps))
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()


@pytest.fixture
def cfr_video(tmp_path: Path) -> Path:
    path = tmp_path / "cfr_30fps.mp4"
    _make_video(path, frame_count=10, fps=FPS_CFR)
    return path


@pytest.fixture
def rotated_video_90(tmp_path: Path) -> Path:
    path = tmp_path / "rotated_90.mp4"
    _make_video(path, frame_count=5, fps=FPS_CFR, rotation=90)
    return path


@pytest.fixture
def vfr_video(tmp_path: Path) -> Path:
    path = tmp_path / "vfr.mp4"
    # 5 frames with durations 33, 50, 33, 66, 33 ms
    _make_video(path, frame_count=5, fps=30, vfr_durations_ms=[33, 50, 33, 66, 33])
    return path

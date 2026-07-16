"""Unit tests for PTS/time utilities."""

from __future__ import annotations

import pytest

from mergenvision_video_lab.time_utils import ns_to_pts, pts_to_ns


def test_pts_to_ns_30fps() -> None:
    # 30 fps: time_base = 1/30, pts=30 -> 1 second = 1_000_000_000 ns
    assert pts_to_ns(30, 1, 30) == 1_000_000_000


def test_pts_to_ns_millisecond_timebase() -> None:
    # time_base = 1/1000 -> pts=1000 is exactly 1 second
    pts_ns = pts_to_ns(1000, 1, 1000)
    assert pts_ns == 1_000_000_000


def test_pts_to_ns_round_trip() -> None:
    tb_num, tb_den = 1, 25
    for pts in [0, 12, 25, 100]:
        ns = pts_to_ns(pts, tb_num, tb_den)
        assert ns_to_pts(ns, tb_num, tb_den) == pts


def test_pts_to_ns_zero_den_raises() -> None:
    with pytest.raises(ValueError):
        pts_to_ns(0, 1, 0)

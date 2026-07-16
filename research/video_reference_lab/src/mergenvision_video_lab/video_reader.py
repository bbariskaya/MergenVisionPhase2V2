"""PyAV-based reference video decoding and probing."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np

from mergenvision_video_lab.errors import VideoReadError
from mergenvision_video_lab.time_utils import pts_to_ns


@dataclass(frozen=True, slots=True)
class VideoProbeResult:
    """Container-level video metadata."""

    container: str
    codec: str
    pixel_format: str
    width: int
    height: int
    display_width: int
    display_height: int
    rotation_degrees: float
    sample_aspect_ratio: tuple[int, int] | None
    display_aspect_ratio: tuple[int, int] | None
    avg_frame_rate: float
    real_frame_rate: float | None
    time_base_num: int
    time_base_den: int
    start_time: int | None
    duration_ns: int | None
    estimated_frame_count: int | None
    audio_stream_count: int
    video_stream_index: int


@dataclass(frozen=True, slots=True)
class VideoFrame:
    """One decoded frame with canonical temporal metadata."""

    source_id: str
    frame_index: int
    pts: int
    time_base_num: int
    time_base_den: int
    pts_ns: int
    width: int
    height: int
    display_width: int
    display_height: int
    rotation_degrees: float
    ndarray: np.ndarray


def _read_rotation(stream: av.video.stream.VideoStream) -> float:
    """Extract rotation from stream side data / metadata."""
    # PyAV exposes side_data as a list of dict-like entries.
    rotation = 0.0
    try:
        for side in stream.side_data or []:
            if getattr(side, "type", None) == "displaymatrix":
                rotation = float(side.rotation)
                break
    except Exception:
        pass
    # Fallback to metadata tags sometimes used by phones.
    if rotation == 0.0:
        rotate_tag = stream.metadata.get("rotate") if stream.metadata else None
        if rotate_tag is not None:
            with contextlib.suppress(ValueError):
                rotation = float(rotate_tag)
    return rotation


def _sar_to_tuple(sar: Any) -> tuple[int, int] | None:
    if sar is None or sar.denominator == 0:
        return None
    return (int(sar.numerator), int(sar.denominator))


def probe_video(path: Path | str) -> VideoProbeResult:
    """Probe video container and first video stream metadata."""
    path = Path(path)
    if not path.exists():
        raise VideoReadError(f"video file not found: {path}")

    try:
        container = av.open(str(path))
    except av.AVError as exc:
        raise VideoReadError(f"cannot open video: {exc}") from exc

    video_stream = None
    audio_stream_count = 0
    for stream in container.streams:
        if stream.type == "video" and video_stream is None:
            video_stream = stream
        elif stream.type == "audio":
            audio_stream_count += 1

    if video_stream is None:
        raise VideoReadError("no video stream found")

    rotation = _read_rotation(video_stream)

    # Stored (coded) vs display dimensions.
    width = int(video_stream.width)
    height = int(video_stream.height)
    if rotation in (90.0, 270.0, -90.0, -270.0):
        display_width, display_height = height, width
    else:
        display_width, display_height = width, height

    avg_rate = video_stream.average_rate
    time_base = video_stream.time_base

    duration = video_stream.duration
    duration_ns = None
    if duration is not None and time_base is not None and time_base.denominator != 0:
        duration_ns = pts_to_ns(int(duration), int(time_base.numerator), int(time_base.denominator))

    estimated_frame_count = None
    if avg_rate is not None and avg_rate.denominator != 0 and duration is not None:
        fps = float(avg_rate)
        duration_sec = duration_ns / 1e9 if duration_ns else None
        if duration_sec is not None and fps > 0:
            estimated_frame_count = int(round(duration_sec * fps))

    start_time = int(video_stream.start_time) if video_stream.start_time is not None else None

    # PyAV does not expose a distinct real_rate attribute on all streams.
    real_frame_rate = None
    if hasattr(video_stream, "real_rate") and video_stream.real_rate is not None:
        rr = video_stream.real_rate
        if rr.denominator != 0:
            real_frame_rate = float(rr)

    result = VideoProbeResult(
        container=Path(path).suffix.lstrip(".").lower() or "unknown",
        codec=video_stream.codec.name if video_stream.codec else "unknown",
        pixel_format=video_stream.pix_fmt or "unknown",
        width=width,
        height=height,
        display_width=display_width,
        display_height=display_height,
        rotation_degrees=rotation,
        sample_aspect_ratio=_sar_to_tuple(video_stream.sample_aspect_ratio),
        display_aspect_ratio=_sar_to_tuple(video_stream.display_aspect_ratio),
        avg_frame_rate=float(avg_rate)
        if avg_rate is not None and avg_rate.denominator != 0
        else 0.0,
        real_frame_rate=real_frame_rate,
        time_base_num=int(time_base.numerator) if time_base is not None else 1,
        time_base_den=int(time_base.denominator) if time_base is not None else 1,
        start_time=start_time,
        duration_ns=duration_ns,
        estimated_frame_count=estimated_frame_count,
        audio_stream_count=audio_stream_count,
        video_stream_index=int(video_stream.index),
    )
    container.close()
    return result


class VideoReader:
    """Sequential PyAV decoder yielding canonical VideoFrame records."""

    def __init__(self, path: Path | str, source_id: str | None = None) -> None:
        self.path = Path(path)
        self.source_id = source_id or self.path.name
        self._probe = probe_video(self.path)
        self._frame_index = 0
        self._last_pts_ns: int | None = None

    @property
    def probe(self) -> VideoProbeResult:
        return self._probe

    def __iter__(self) -> Iterator[VideoFrame]:
        try:
            container = av.open(str(self.path))
        except av.AVError as exc:
            raise VideoReadError(f"cannot open video: {exc}") from exc

        stream = container.streams.video[self._probe.video_stream_index]
        tb_num = self._probe.time_base_num
        tb_den = self._probe.time_base_den

        for frame in container.decode(stream):
            pts = frame.pts
            if pts is None:
                raise VideoReadError(
                    "missing PTS in strict decode mode",
                    {"frame_index": self._frame_index},
                )
            pts_ns = pts_to_ns(int(pts), tb_num, tb_den)

            if self._last_pts_ns is not None and pts_ns < self._last_pts_ns:
                raise VideoReadError(
                    "PTS regression",
                    {
                        "frame_index": self._frame_index,
                        "pts_ns": pts_ns,
                        "last_pts_ns": self._last_pts_ns,
                    },
                )

            ndarray = frame.to_ndarray(format="bgr24")
            yield VideoFrame(
                source_id=self.source_id,
                frame_index=self._frame_index,
                pts=int(pts),
                time_base_num=tb_num,
                time_base_den=tb_den,
                pts_ns=pts_ns,
                width=self._probe.width,
                height=self._probe.height,
                display_width=self._probe.display_width,
                display_height=self._probe.display_height,
                rotation_degrees=self._probe.rotation_degrees,
                ndarray=ndarray,
            )
            self._frame_index += 1
            self._last_pts_ns = pts_ns

        container.close()

    def reset(self) -> None:
        """Reset iteration state so the reader can be reused."""
        self._frame_index = 0
        self._last_pts_ns = None

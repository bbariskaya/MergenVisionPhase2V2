"""FFprobe-based video media probing."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.domain.errors import InvalidMediaError, PayloadTooLargeError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeResult:
    container_format: str
    video_codec: str
    pixel_format: str | None
    display_width: int
    display_height: int
    rotation_degrees: int
    duration_ns: int
    time_base_num: int
    time_base_den: int
    nominal_fps_num: int | None
    nominal_fps_den: int | None
    total_frames: int | None


def _split_rational(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    parts = value.split("/")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _seconds_to_ns(seconds: float) -> int:
    return int(round(seconds * 1_000_000_000.0))


def _map_container(raw: str | None) -> str:
    if not raw:
        return "unknown"
    first = raw.split(",")[0].strip().lower()
    if first in ("mov", "mp4", "m4a", "3gp", "3g2", "mj2"):
        return "mp4"
    if first in ("matroska", "webm"):
        return "mkv"
    return first


def _video_stream_rotation(stream: dict[str, Any]) -> int:
    tags = stream.get("tags") or {}
    for key in ("rotate", "rotation"):
        raw = tags.get(key)
        if raw:
            try:
                return int(float(raw)) % 360
            except ValueError:
                return 0
    side_data_list = stream.get("side_data_list") or []
    for side in side_data_list:
        raw = side.get("rotation")
        if raw is not None:
            try:
                return int(float(raw)) % 360
            except ValueError:
                return 0
    return 0


async def probe_video(
    path: Path,
    ffprobe_command: list[str],
    timeout_seconds: float = 30.0,
    max_duration_ns: int | None = None,
) -> ProbeResult:
    cmd = list(ffprobe_command)
    cmd.extend(
        [
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-print_format",
            "json",
            str(path),
        ]
    )

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout_seconds,
        )
    except TimeoutError as exc:
        raise PayloadTooLargeError("Video probe timed out") from exc
    except FileNotFoundError as exc:
        raise InvalidMediaError("ffprobe not available") from exc

    if proc.returncode != 0:
        message = (stderr.decode("utf-8", errors="replace") or "ffprobe failed").strip()
        raise InvalidMediaError(f"Video probe failed: {message}")

    try:
        data = json.loads(stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise InvalidMediaError("Video probe returned invalid JSON") from exc

    streams = data.get("streams") or []
    video_stream = next(
        (s for s in streams if s.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise InvalidMediaError("No video stream found")

    width = video_stream.get("width")
    height = video_stream.get("height")
    if not width or not height:
        raise InvalidMediaError("Video stream missing width/height")

    codec = (video_stream.get("codec_name") or "").lower()
    if codec not in {"h264", "hevc", "mpeg4"}:
        raise InvalidMediaError(f"Unsupported video codec: {codec}")

    container_raw = (data.get("format") or {}).get("format_name")
    container_format = _map_container(container_raw)

    duration_seconds: float | None = None
    duration_raw = video_stream.get("duration")
    if duration_raw is None:
        duration_raw = (data.get("format") or {}).get("duration")
    if duration_raw is not None:
        try:
            duration_seconds = float(duration_raw)
        except ValueError:
            duration_seconds = None
    if duration_seconds is None or duration_seconds < 0:
        raise InvalidMediaError("Video duration unavailable")

    duration_ns = _seconds_to_ns(duration_seconds)
    if max_duration_ns is not None and duration_ns > max_duration_ns:
        raise PayloadTooLargeError("Video exceeds maximum allowed duration")

    time_base = _split_rational(video_stream.get("time_base"))
    if time_base is None:
        time_base = (1, 1_000_000_000)
    time_base_num, time_base_den = time_base

    fps = _split_rational(video_stream.get("r_frame_rate"))
    if fps is None:
        fps = _split_rational(video_stream.get("avg_frame_rate"))

    total_frames_raw = video_stream.get("nb_frames")
    total_frames = None
    if total_frames_raw is not None:
        try:
            total_frames = int(total_frames_raw)
            if total_frames < 0:
                total_frames = None
        except ValueError:
            total_frames = None

    pixel_format = video_stream.get("pix_fmt")
    rotation_degrees = _video_stream_rotation(video_stream)

    return ProbeResult(
        container_format=container_format,
        video_codec=codec,
        pixel_format=pixel_format,
        display_width=int(width),
        display_height=int(height),
        rotation_degrees=rotation_degrees,
        duration_ns=duration_ns,
        time_base_num=time_base_num,
        time_base_den=time_base_den,
        nominal_fps_num=fps[0] if fps else None,
        nominal_fps_den=fps[1] if fps else None,
        total_frames=total_frames,
    )

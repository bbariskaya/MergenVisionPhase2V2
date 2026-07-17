from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import zstandard
from google.protobuf.internal.decoder import _DecodeVarint32  # type: ignore[import-untyped]
from google.protobuf.message import DecodeError  # type: ignore[import-untyped]

from app.infrastructure.serialization import (
    RawTrackTemplate,
    TrackTemplateBundle,
    TrackTemplateFooter,
)


class TrackTemplateArtifactError(Exception):
    pass


_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def read_track_templates(artifact_path: Path) -> list[RawTrackTemplate]:
    data = artifact_path.read_bytes()
    if data.startswith(_ZSTD_MAGIC):
        data = zstandard.ZstdDecompressor().decompress(data)
    return list(_iter_templates(io.BytesIO(data)))


def _iter_templates(stream: BinaryIO) -> list[RawTrackTemplate]:
    buf = stream.read()
    pos = 0
    messages: list[bytes] = []

    while pos < len(buf):
        msg_len, new_pos = _DecodeVarint32(buf, pos)
        pos = new_pos
        if pos + msg_len > len(buf):
            raise TrackTemplateArtifactError("truncated message length")
        messages.append(buf[pos : pos + msg_len])
        pos += msg_len

    if not messages:
        raise TrackTemplateArtifactError("empty template artifact")

    bundle = TrackTemplateBundle()
    try:
        bundle_parsed = bundle.ParseFromString(messages[0])
    except DecodeError:
        bundle_parsed = 0
    if not bundle_parsed:
        raise TrackTemplateArtifactError("missing TrackTemplateBundle header")

    footer: TrackTemplateFooter | None = None
    if len(messages) > 1:
        footer = TrackTemplateFooter()
        try:
            footer_parsed = footer.ParseFromString(messages[-1])
        except DecodeError:
            footer_parsed = 0
        if not footer_parsed:
            raise TrackTemplateArtifactError("invalid TrackTemplateFooter")

    template_messages = messages[1:-1] if footer is not None else messages[1:]
    templates: list[RawTrackTemplate] = []
    for msg_bytes in template_messages:
        t = RawTrackTemplate()
        try:
            template_parsed = t.ParseFromString(msg_bytes)
        except DecodeError:
            template_parsed = 0
        if not template_parsed:
            raise TrackTemplateArtifactError("invalid RawTrackTemplate")
        templates.append(t)

    if footer is not None and footer.template_count != len(templates):
        raise TrackTemplateArtifactError(
            f"footer template_count {footer.template_count} != read {len(templates)}"
        )

    return templates

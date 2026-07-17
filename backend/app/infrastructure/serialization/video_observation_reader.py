from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import zstandard
from google.protobuf.internal.decoder import _DecodeVarint32  # type: ignore[import-untyped]

from app.application.ports.video_observations import FaceObservation, VideoObservationFrame
from app.domain.value_objects import BoundingBox
from app.infrastructure.serialization import (
    FaceDetection as FaceDetectionProto,
)
from app.infrastructure.serialization import (
    ObservationChunkFooter,
)
from app.infrastructure.serialization import (
    VideoObservationFrame as VideoObservationFrameProto,
)


def _proto_to_face_observation(proto: FaceDetectionProto) -> FaceObservation:
    return FaceObservation(
        detection_id=proto.detection_id,
        ordinal=proto.ordinal,
        bbox=BoundingBox(
            x=proto.x,
            y=proto.y,
            width=proto.width,
            height=proto.height,
        ),
        landmarks=tuple(proto.landmarks),
        detector_score=proto.detector_score,
        quality_score=proto.quality_score,
        tracking_eligible=proto.tracking_eligible,
        recognition_eligible=proto.recognition_eligible,
        rejection_code=proto.rejection_code,
        embedding=tuple(proto.embedding) if proto.embedding else (),
        model_version=proto.model_version,
        preprocess_version=proto.preprocess_version,
        raw_track_key=proto.raw_track_key,
    )


def _proto_to_frame(proto: VideoObservationFrameProto) -> VideoObservationFrame:
    return VideoObservationFrame(
        job_id=proto.job_id,
        video_id=proto.video_id,
        stream_index=proto.stream_index,
        frame_index=proto.frame_index,
        source_pts=proto.source_pts,
        pts_ns=proto.pts_ns,
        display_width=proto.display_width,
        display_height=proto.display_height,
        detections=tuple(_proto_to_face_observation(d) for d in proto.detections),
    )


class ObservationArtifactError(Exception):
    pass


_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def read_observation_frames(artifact_path: Path) -> list[VideoObservationFrame]:
    data = artifact_path.read_bytes()
    if data.startswith(_ZSTD_MAGIC):
        data = zstandard.ZstdDecompressor().decompress(data)
    return list(_iter_frames(io.BytesIO(data)))


def _iter_frames(stream: BinaryIO) -> list[VideoObservationFrame]:
    frames: list[VideoObservationFrame] = []
    buf = stream.read()
    pos = 0
    footer: ObservationChunkFooter | None = None
    while pos < len(buf):
        msg_len, new_pos = _DecodeVarint32(buf, pos)
        pos = new_pos
        if pos + msg_len > len(buf):
            raise ObservationArtifactError("truncated message length")
        msg_bytes = buf[pos : pos + msg_len]
        pos += msg_len

        frame = VideoObservationFrameProto()
        if frame.ParseFromString(msg_bytes):
            frames.append(_proto_to_frame(frame))
            continue

        footer = ObservationChunkFooter()
        if not footer.ParseFromString(msg_bytes):
            raise ObservationArtifactError("invalid protobuf message")

    if footer is not None and footer.frame_count != len(frames):
        raise ObservationArtifactError(
            f"footer frame_count {footer.frame_count} != read {len(frames)}"
        )
    return frames

import io
from pathlib import Path
from uuid import uuid4

import zstandard

from app.application.ports.video_observations import VideoObservationFrame
from app.domain.value_objects import BoundingBox
from app.infrastructure.serialization import VideoObservationFrame as VideoObservationFrameProto
from app.infrastructure.serialization.video_observation_reader import read_observation_frames


def _varint_bytes(value: int) -> bytes:
    from google.protobuf.internal.encoder import _EncodeVarint

    out = io.BytesIO()
    _EncodeVarint(out.write, value)
    return out.getvalue()


def _build_proto_frame(job_id: str, frame_index: int) -> VideoObservationFrameProto:
    proto = VideoObservationFrameProto()
    proto.job_id = job_id
    proto.video_id = str(uuid4())
    proto.frame_index = frame_index
    proto.pts_ns = frame_index * 33_000_000
    proto.display_width = 640
    proto.display_height = 480
    det = proto.detections.add()
    det.detection_id = f"{job_id}:{frame_index}:0"
    det.ordinal = 0
    det.x = 100
    det.y = 100
    det.width = 80
    det.height = 90
    det.landmarks.extend([10.0] * 10)
    det.detector_score = 0.95
    det.quality_score = 0.9
    det.tracking_eligible = True
    det.recognition_eligible = True
    det.rejection_code = ""
    det.embedding.extend([0.0] * 512)
    det.model_version = "retinaface_r50_glintr100_v1"
    det.preprocess_version = "cuda_five_point_align"
    return proto


def test_read_uncompressed_frames(tmp_path: Path) -> None:
    job_id = str(uuid4())
    path = tmp_path / "obs.0.pb"
    with path.open("wb") as f:
        for i in range(3):
            proto = _build_proto_frame(job_id, i)
            payload = proto.SerializeToString()
            f.write(_varint_bytes(len(payload)) + payload)

    frames = read_observation_frames(path)
    assert len(frames) == 3
    assert all(isinstance(f, VideoObservationFrame) for f in frames)
    assert frames[0].detections[0].bbox == BoundingBox(x=100, y=100, width=80, height=90)


def test_read_zstd_compressed_frames(tmp_path: Path) -> None:
    job_id = str(uuid4())
    raw = io.BytesIO()
    for i in range(2):
        proto = _build_proto_frame(job_id, i)
        payload = proto.SerializeToString()
        raw.write(_varint_bytes(len(payload)) + payload)
    compressed = zstandard.ZstdCompressor().compress(raw.getvalue())
    path = tmp_path / "obs.0.pb.zst"
    path.write_bytes(compressed)

    frames = read_observation_frames(path)
    assert len(frames) == 2

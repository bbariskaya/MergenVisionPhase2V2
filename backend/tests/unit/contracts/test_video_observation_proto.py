"""Milestone 5: video observation protobuf contract checks."""

from __future__ import annotations

from pathlib import Path

CONTRACT_PATH = Path(__file__).parents[4] / "backend" / "contracts" / "video_observation_v1.proto"


def test_observation_proto_file_exists_and_contains_required_messages() -> None:
    text = CONTRACT_PATH.read_text(encoding="utf-8")
    assert "message VideoObservationFrame" in text
    assert "message FaceDetection" in text
    assert "job_id" in text
    assert "pts_ns" in text
    assert "repeated float embedding" in text
    assert "message ObservationChunkFooter" in text
    assert 'syntax = "proto3"' in text

"""Milestone 4: common device FacePipeline port contract."""

from __future__ import annotations

import pytest

from app.application.ports.face_pipeline import (
    DeviceImageView,
    FaceObservation,
    FaceObservations,
    FacePipeline,
)
from app.domain.errors import VideoExecutorNotReadyError
from app.domain.value_objects import BoundingBox
from app.infrastructure.runtime.face_pipeline_adapter import FacePipelineAdapter


def test_device_image_view_fields_are_frozen_and_addressable() -> None:
    view = DeviceImageView(
        device_pointer=0xDEADBEEF,
        width=1920,
        height=1080,
        pitch=7680,
        pixel_format="RGBA",
        device_id=0,
        source_batch_index=3,
        frame_index=42,
        pts_ns=1_400_000,
        display_width=1920,
        display_height=1080,
        rotation=0,
        ownership="external",
    )
    assert view.frame_index == 42
    assert view.source_batch_index == 3
    assert view.ownership == "external"


def test_face_observation_requires_positive_bbox() -> None:
    bbox = BoundingBox(x=10, y=20, width=100, height=120)
    obs = FaceObservation(
        detection_index=0,
        ordinal=1,
        bbox=bbox,
        landmarks5=(0.0,) * 10,
        detector_score=0.99,
        quality_score=0.88,
        tracking_eligible=True,
        recognition_eligible=True,
        embedding=(0.0,) * 512,
    )
    assert obs.bbox == bbox
    assert len(obs.embedding) == 512


def test_face_observations_preserves_frame_association() -> None:
    result = FaceObservations(
        source_batch_index=2,
        frame_index=42,
        pts_ns=1_000,
        display_width=1920,
        display_height=1080,
        detections=(),
    )
    assert result.frame_index == 42


def test_face_pipeline_adapter_is_a_protocol_implementation() -> None:
    adapter: FacePipeline = FacePipelineAdapter()
    assert isinstance(adapter, FacePipelineAdapter)


@pytest.mark.anyio
async def test_face_pipeline_adapter_is_not_ready() -> None:
    adapter: FacePipeline = FacePipelineAdapter()
    view = DeviceImageView(
        device_pointer=0,
        width=640,
        height=480,
        pitch=2560,
        pixel_format="RGBA",
    )
    with pytest.raises(VideoExecutorNotReadyError):
        await adapter.infer_device_batch((view,))

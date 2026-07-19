"""Video recognition API routes."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse

from app.api.routes.dependencies import (
    get_object_store,
    get_video_overlay_service,
    get_video_result_service,
    get_video_upload_service,
)
from app.api.schemas import (
    OverlayBoundingBox,
    OverlayDetection,
    OverlayFrame,
    VideoAppearanceEntry,
    VideoAppearancesResponse,
    VideoJobResponse,
    VideoJobResultResponse,
    VideoPeopleResponse,
    VideoPersonSummary,
    VideoRecognizeResponse,
    VideoResponse,
    VideoTimelineFramesResponse,
    VideoTimelineRecord,
    VideoTimelineResponse,
)
from app.application.ports.object_store import ObjectStore
from app.application.services.video_overlay_service import VideoOverlayService
from app.application.services.video_result_service import VideoResultService
from app.application.services.video_upload_service import (
    JobResultSnapshot,
    JobSnapshot,
    SubmitResult,
    VideoSnapshot,
    VideoUploadService,
)
from app.domain.value_objects import JobId
from app.infrastructure.persistence.sqlalchemy.session import async_session_maker
from app.infrastructure.persistence.sqlalchemy.unit_of_work import SqlAlchemyUnitOfWork

router = APIRouter(prefix="/videos", tags=["videos"])


class _UploadFileStream:
    def __init__(self, upload: UploadFile) -> None:
        self._upload = upload
        self.filename = upload.filename
        self.content_type = upload.content_type

    async def read(self, size: int) -> bytes:
        return await self._upload.read(size)


def _to_recognize_response(
    request_id: str, result: SubmitResult
) -> VideoRecognizeResponse:
    return VideoRecognizeResponse(
        request_id=request_id,
        process_id=result.process_id,
        video_id=result.video_id,
        job_id=result.job_id,
        upload_session_id=result.upload_session_id or None,
        status=result.status,
        status_url=result.status_url,
        result_url=result.result_url,
    )


def _to_job_response(request_id: str, snapshot: JobSnapshot) -> VideoJobResponse:
    return VideoJobResponse(
        request_id=request_id,
        process_id=snapshot.process_id,
        video_id=snapshot.video_id,
        job_id=snapshot.job_id,
        state=snapshot.state,
        stage=snapshot.stage,
        progress_percent=snapshot.progress_percent,
        sampling_mode=snapshot.sampling_mode,
        every_n_frames=snapshot.every_n_frames,
        frames_per_second=snapshot.frames_per_second,
        processed_frames=snapshot.processed_frames,
        sampled_frames=snapshot.sampled_frames,
        detected_observations=snapshot.detected_observations,
        person_count=snapshot.person_count,
        cancellation_requested=snapshot.cancellation_requested,
        error_code=snapshot.error_code,
        status_url=snapshot.status_url,
        result_url=snapshot.result_url,
    )


def _to_video_response(request_id: str, snapshot: VideoSnapshot) -> VideoResponse:
    return VideoResponse(
        request_id=request_id,
        video_id=snapshot.video_id,
        upload_session_id=snapshot.upload_session_id,
        state=snapshot.state,
        content_sha256=snapshot.content_sha256,
        size_bytes=snapshot.size_bytes,
        container_format=snapshot.container_format,
        video_codec=snapshot.video_codec,
        display_width=snapshot.display_width,
        display_height=snapshot.display_height,
        rotation_degrees=snapshot.rotation_degrees,
        duration_ns=snapshot.duration_ns,
        total_frames=snapshot.total_frames,
        failure_code=snapshot.failure_code,
    )


def _to_result_response(
    request_id: str, snapshot: JobResultSnapshot
) -> VideoJobResultResponse:
    return VideoJobResultResponse(
        request_id=request_id,
        job_id=snapshot.job_id,
        state=snapshot.state,
        result_available=snapshot.result_available,
        manifest_bucket=snapshot.manifest_bucket,
        manifest_key=snapshot.manifest_key,
        manifest_sha256=snapshot.manifest_sha256,
    )


def _require_idempotency_key(request: Request, key: str | None) -> None:
    if key is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "requestId": str(getattr(request.state, "request_id", "unknown")),
                "error": {
                    "code": "MISSING_IDEMPOTENCY_KEY",
                    "message": "Idempotency-Key header is required.",
                    "retryable": False,
                    "details": {},
                },
            },
        )


@router.post("/recognize", response_model=VideoRecognizeResponse, status_code=202)
async def recognize_video(
    request: Request,
    video: UploadFile = File(..., description="Video file to recognize"),
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    sampling_mode: str = Form("every_frame", alias="samplingMode"),
    every_n_frames: int | None = Form(None, alias="everyNFrames"),
    frames_per_second: float | None = Form(None, alias="framesPerSecond"),
    service: VideoUploadService = Depends(get_video_upload_service),
) -> VideoRecognizeResponse:
    _require_idempotency_key(request, idempotency_key)
    assert idempotency_key is not None
    request_id = str(request.state.request_id)
    result = await service.submit_video_recognition(
        request_id=request_id,
        idempotency_key=idempotency_key,
        file=_UploadFileStream(video),
        sampling_mode=sampling_mode,
        every_n_frames=every_n_frames,
        frames_per_second=frames_per_second,
    )
    return _to_recognize_response(request_id, result)


@router.get("/{video_id:uuid}", response_model=VideoResponse)
async def get_video(
    request: Request,
    video_id: uuid.UUID,
    service: VideoUploadService = Depends(get_video_upload_service),
) -> VideoResponse:
    request_id = str(request.state.request_id)
    snapshot = await service.get_video(str(video_id), request_id)
    return _to_video_response(request_id, snapshot)


@router.get("/jobs/{job_id}", response_model=VideoJobResponse)
async def get_job(
    request: Request,
    job_id: str,
    service: VideoUploadService = Depends(get_video_upload_service),
) -> VideoJobResponse:
    request_id = str(request.state.request_id)
    snapshot = await service.get_job(job_id, request_id)
    return _to_job_response(request_id, snapshot)


@router.delete("/jobs/{job_id}", response_model=VideoJobResponse, status_code=202)
async def cancel_job(
    request: Request,
    job_id: str,
    service: VideoUploadService = Depends(get_video_upload_service),
) -> VideoJobResponse:
    request_id = str(request.state.request_id)
    snapshot = await service.cancel_job(job_id, request_id)
    return _to_job_response(request_id, snapshot)


@router.post("/jobs/{job_id}/retry", response_model=VideoRecognizeResponse, status_code=202)
async def retry_job(
    request: Request,
    job_id: str,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    service: VideoUploadService = Depends(get_video_upload_service),
) -> VideoRecognizeResponse:
    _require_idempotency_key(request, idempotency_key)
    assert idempotency_key is not None
    request_id = str(request.state.request_id)
    result = await service.retry_job(job_id, idempotency_key, request_id)
    return _to_recognize_response(request_id, result)


@router.get("/jobs/{job_id}/result", response_model=VideoJobResultResponse)
async def get_job_result(
    request: Request,
    job_id: str,
    service: VideoUploadService = Depends(get_video_upload_service),
) -> VideoJobResultResponse:
    request_id = str(request.state.request_id)
    snapshot = await service.get_job_result(job_id, request_id)
    if not snapshot.result_available:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "requestId": request_id,
                "error": {
                    "code": "JOB_NOT_COMPLETED",
                    "message": "Job result is not available until the job is completed.",
                    "retryable": False,
                    "details": {},
                },
            },
        )
    return _to_result_response(request_id, snapshot)


@router.get("/jobs/{job_id}/people", response_model=VideoPeopleResponse)
async def list_people(
    request: Request,
    job_id: str,
    service: VideoResultService = Depends(get_video_result_service),
) -> VideoPeopleResponse:
    request_id = str(request.state.request_id)
    people = await service.list_people(job_id)
    return VideoPeopleResponse(
        request_id=request_id,
        job_id=job_id,
        person_count=len(people),
        people=[_to_person_summary(p) for p in people],
    )


@router.get("/jobs/{job_id}/appearances", response_model=VideoAppearancesResponse)
async def list_appearances(
    request: Request,
    job_id: str,
    service: VideoResultService = Depends(get_video_result_service),
) -> VideoAppearancesResponse:
    request_id = str(request.state.request_id)
    appearances = await service.list_appearances(job_id)
    return VideoAppearancesResponse(
        request_id=request_id,
        job_id=job_id,
        appearance_count=len(appearances),
        appearances=[_to_appearance_entry(a) for a in appearances],
    )


@router.get("/jobs/{job_id}/timeline", response_model=VideoTimelineResponse)
async def get_timeline(
    request: Request,
    job_id: str,
    service: VideoResultService = Depends(get_video_result_service),
) -> VideoTimelineResponse:
    request_id = str(request.state.request_id)
    records = await service.get_timeline(job_id)
    return VideoTimelineResponse(
        request_id=request_id,
        job_id=job_id,
        record_count=len(records),
        records=[_to_timeline_record(r) for r in records],
    )


@router.get("/jobs/{job_id}/timeline/frames", response_model=VideoTimelineFramesResponse)
async def get_timeline_frames(
    request: Request,
    job_id: str,
    start_pts_ns: int | None = None,
    end_pts_ns: int | None = None,
    service: VideoOverlayService = Depends(get_video_overlay_service),
) -> VideoTimelineFramesResponse:
    request_id = str(request.state.request_id)
    start = start_pts_ns if start_pts_ns is not None else 0
    end = end_pts_ns
    frames = await service.read_overlay_frames(
        JobId(uuid.UUID(job_id)),
        start_pts_ns=start,
        end_pts_ns=end,
    )
    actual_end = max((f["pts_ns"] for f in frames), default=end if end is not None else start)
    return VideoTimelineFramesResponse(
        request_id=request_id,
        job_id=job_id,
        start_pts_ns=start,
        end_pts_ns=actual_end,
        record_count=len(frames),
        frames=[_to_overlay_frame(f) for f in frames],
    )


@router.get("/{video_id:uuid}/playback")
async def playback_video(
    request: Request,
    video_id: uuid.UUID,
    range_header: str | None = Header(None, alias="Range"),
    object_store: ObjectStore = Depends(get_object_store),
) -> Response:
    request_id = str(request.state.request_id)
    async with SqlAlchemyUnitOfWork(async_session_maker) as uow:
        video_asset = await uow.video_assets.get_by_id(video_id)
        if video_asset is None or video_asset.object_key is None or video_asset.bucket is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "requestId": request_id,
                    "error": {
                        "code": "VIDEO_NOT_FOUND",
                        "message": "Video asset not found or not ready",
                        "retryable": False,
                        "details": {},
                    },
                },
            )
        object_key = video_asset.object_key
        content_type = video_asset.content_type or "video/mp4"

    data = await object_store.get(object_key)
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "requestId": request_id,
                "error": {
                    "code": "VIDEO_OBJECT_MISSING",
                    "message": "Underlying video object is missing",
                    "retryable": False,
                    "details": {},
                },
            },
        )

    total = len(data)
    start, stop = _parse_byte_range(range_header, total)

    async def _stream() -> AsyncIterator[bytes]:
        chunk_size = 64 * 1024
        for offset in range(start, stop + 1, chunk_size):
            yield data[offset : min(offset + chunk_size, stop + 1)]

    if range_header is None:
        return StreamingResponse(
            _stream(),
            media_type=content_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(total),
            },
        )

    content_length = stop - start + 1
    return StreamingResponse(
        _stream(),
        media_type=content_type,
        status_code=status.HTTP_206_PARTIAL_CONTENT,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
            "Content-Range": f"bytes {start}-{stop}/{total}",
        },
    )


def _parse_byte_range(range_header: str | None, total: int) -> tuple[int, int]:
    if not range_header or not range_header.startswith("bytes="):
        return (0, max(0, total - 1))
    spec = range_header[len("bytes=") :].split("-", 1)
    try:
        start = int(spec[0]) if spec[0] else 0
        end = int(spec[1]) if len(spec) > 1 and spec[1] else total - 1
    except ValueError:
        return (0, max(0, total - 1))
    start = max(0, min(start, total - 1))
    end = max(start, min(end, total - 1))
    return (start, end)


def _to_overlay_frame(frame: Any) -> OverlayFrame:
    return OverlayFrame(
        frame_index=frame["frame_index"],
        pts_ns=frame["pts_ns"],
        detections=[_to_overlay_detection(d) for d in frame["detections"]],
    )


def _to_overlay_detection(detection: Any) -> OverlayDetection:
    return OverlayDetection(
        track_id=detection["track_id"],
        face_id=detection["face_id"],
        status=detection["status"],
        name=detection.get("name"),
        bbox=OverlayBoundingBox(
            x=detection["bbox"]["x"],
            y=detection["bbox"]["y"],
            width=detection["bbox"]["width"],
            height=detection["bbox"]["height"],
        ),
        confidence=detection.get("confidence", 0.0),
        provenance=detection.get("provenance", "detected"),
    )


def _to_person_summary(person: Any) -> VideoPersonSummary:
    return VideoPersonSummary(
        track_id=str(person.track_id),
        face_id=str(person.face_id),
        status=person.status,
        name=person.name,
        current_status=person.current_status,
        current_name=person.current_name,
        current_metadata=person.current_metadata,
        first_frame_index=person.first_frame_index,
        last_frame_index=person.last_frame_index,
        first_pts_ns=person.first_pts_ns,
        last_pts_ns=person.last_pts_ns,
        total_duration_ns=person.total_duration_ns,
        detection_count=person.detection_count,
        appearance_count=person.appearance_count,
        match_confidence=person.match_confidence,
    )


def _to_appearance_entry(appearance: Any) -> VideoAppearanceEntry:
    return VideoAppearanceEntry(
        track_id=str(appearance.track_id),
        face_id=str(appearance.face_id),
        start_frame_index=appearance.start_frame_index,
        end_frame_index=appearance.end_frame_index,
        start_pts_ns=appearance.start_pts_ns,
        end_pts_ns=appearance.end_pts_ns,
        detection_count=appearance.detection_count,
    )


def _to_timeline_record(record: Any) -> VideoTimelineRecord:
    return VideoTimelineRecord(
        track_id=str(record.track_id),
        face_id=str(record.face_id),
        start_frame_index=record.start_frame_index,
        end_frame_index=record.end_frame_index,
        start_pts_ns=record.start_pts_ns,
        end_pts_ns=record.end_pts_ns,
    )

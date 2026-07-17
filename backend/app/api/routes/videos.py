"""Video recognition API routes."""

from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    status,
)

from app.api.routes.dependencies import get_video_upload_service
from app.api.schemas import (
    VideoJobResponse,
    VideoJobResultResponse,
    VideoRecognizeResponse,
    VideoResponse,
)
from app.application.services.video_upload_service import (
    JobResultSnapshot,
    JobSnapshot,
    SubmitResult,
    VideoSnapshot,
    VideoUploadService,
)

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


@router.get("/{video_id}", response_model=VideoResponse)
async def get_video(
    request: Request,
    video_id: str,
    service: VideoUploadService = Depends(get_video_upload_service),
) -> VideoResponse:
    request_id = str(request.state.request_id)
    snapshot = await service.get_video(video_id, request_id)
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

"""M2 integration tests: video upload, finalization and async job API."""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.infrastructure.uuid7 import generate_uuid7


def _generate_video(
    path: Path,
    *,
    duration: float = 1.0,
    width: int = 320,
    height: int = 240,
    framerate: int = 30,
    codec: str = "libx264",
    container: str | None = None,
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration}:size={width}x{height}:rate={framerate}",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        codec,
        "-an",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    if container is not None and container != path.suffix.lstrip(".").lower():
        converted = path.with_suffix(f".{container}")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-c:v", "copy", "-f", container, str(converted)],
            check=True,
            capture_output=True,
            text=True,
        )
        path.write_bytes(converted.read_bytes())


@pytest.fixture(scope="session")
def valid_video_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("videos") / "valid.mp4"
    _generate_video(path)
    return path


@pytest.fixture(scope="session")
def unsupported_container_video_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("videos") / "unsupported.avi"
    _generate_video(path, container="avi")
    return path


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def idempotency_key() -> str:
    return f"test-key-{generate_uuid7()}"


def _upload(
    client: TestClient, path: Path, key: str, extra: dict[str, str] | None = None
) -> TestClient.response:
    with path.open("rb") as f:
        response = client.post(
            "/api/v1/videos/recognize",
            files={"video": ("video.mp4", f, "video/mp4")},
            headers={"Idempotency-Key": key},
            data=extra or {},
        )
    return response


def test_submit_video_recognition_returns_202(
    client: TestClient,
    valid_video_file: Path,
    idempotency_key: str,
) -> None:
    response = _upload(client, valid_video_file, idempotency_key)
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["requestId"]
    assert body["processId"]
    assert body["videoId"]
    assert body["jobId"]
    assert body["status"] == "pending"
    assert body["statusUrl"].endswith(f"/api/v1/videos/jobs/{body['jobId']}")
    assert body["resultUrl"].endswith(f"/api/v1/videos/jobs/{body['jobId']}/result")

    job_response = client.get(f"/api/v1/videos/jobs/{body['jobId']}")
    assert job_response.status_code == 200, job_response.text
    job_body = job_response.json()
    assert job_body["state"] == "pending"
    assert job_body["stage"] == "queued"
    assert job_body["processId"] == body["processId"]
    assert job_body["videoId"] == body["videoId"]

    video_response = client.get(f"/api/v1/videos/{body['videoId']}")
    assert video_response.status_code == 200, video_response.text
    video_body = video_response.json()
    assert video_body["state"] == "ready"
    assert video_body["containerFormat"] == "mp4"
    assert video_body["videoCodec"] == "h264"
    assert video_body["sizeBytes"] == valid_video_file.stat().st_size


def test_submit_without_idempotency_key_returns_400(
    client: TestClient,
    valid_video_file: Path,
) -> None:
    with valid_video_file.open("rb") as f:
        response = client.post(
            "/api/v1/videos/recognize",
            files={"video": ("video.mp4", f, "video/mp4")},
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "MISSING_IDEMPOTENCY_KEY"


def test_idempotency_replay_returns_same_ids(
    client: TestClient,
    valid_video_file: Path,
    idempotency_key: str,
) -> None:
    first = _upload(client, valid_video_file, idempotency_key)
    assert first.status_code == 202
    first_body = first.json()

    second = _upload(client, valid_video_file, idempotency_key)
    assert second.status_code == 202
    second_body = second.json()

    assert second_body["videoId"] == first_body["videoId"]
    assert second_body["jobId"] == first_body["jobId"]
    assert second_body["processId"] == first_body["processId"]


def test_idempotency_conflict_for_different_config(
    client: TestClient,
    valid_video_file: Path,
    idempotency_key: str,
) -> None:
    first = _upload(client, valid_video_file, idempotency_key)
    assert first.status_code == 202

    second = _upload(
        client,
        valid_video_file,
        idempotency_key,
        extra={"samplingMode": "every_n_frames", "everyNFrames": "5"},
    )
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"


def test_corrupt_video_is_rejected(
    client: TestClient,
    idempotency_key: str,
) -> None:
    fake_video = io.BytesIO(b"not a video file")
    response = client.post(
        "/api/v1/videos/recognize",
        files={"video": ("video.mp4", fake_video, "video/mp4")},
        headers={"Idempotency-Key": idempotency_key},
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "INVALID_MEDIA"


def test_unsupported_container_is_rejected(
    client: TestClient,
    unsupported_container_video_file: Path,
    idempotency_key: str,
) -> None:
    response = _upload(client, unsupported_container_video_file, idempotency_key)
    assert response.status_code == 415, response.text
    assert response.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_cancel_pending_job(
    client: TestClient,
    valid_video_file: Path,
    idempotency_key: str,
) -> None:
    submit = _upload(client, valid_video_file, idempotency_key)
    assert submit.status_code == 202
    job_id = submit.json()["jobId"]

    cancel = client.delete(f"/api/v1/videos/jobs/{job_id}")
    assert cancel.status_code == 202, cancel.text
    cancel_body = cancel.json()
    assert cancel_body["state"] == "cancelled"


def test_retry_cancelled_job(
    client: TestClient,
    valid_video_file: Path,
    idempotency_key: str,
) -> None:
    submit = _upload(client, valid_video_file, idempotency_key)
    assert submit.status_code == 202
    original_job_id = submit.json()["jobId"]

    cancel = client.delete(f"/api/v1/videos/jobs/{original_job_id}")
    assert cancel.status_code == 202

    retry_key = f"retry-key-{generate_uuid7()}"
    retry = client.post(
        f"/api/v1/videos/jobs/{original_job_id}/retry",
        headers={"Idempotency-Key": retry_key},
    )
    assert retry.status_code == 202, retry.text
    retry_body = retry.json()
    assert retry_body["jobId"] != original_job_id
    assert retry_body["status"] == "pending"

    original = client.get(f"/api/v1/videos/jobs/{original_job_id}")
    assert original.status_code == 200
    assert original.json()["state"] == "cancelled"


def test_get_result_before_completion_returns_409(
    client: TestClient,
    valid_video_file: Path,
    idempotency_key: str,
) -> None:
    submit = _upload(client, valid_video_file, idempotency_key)
    assert submit.status_code == 202
    job_id = submit.json()["jobId"]

    result = client.get(f"/api/v1/videos/jobs/{job_id}/result")
    assert result.status_code == 409, result.text

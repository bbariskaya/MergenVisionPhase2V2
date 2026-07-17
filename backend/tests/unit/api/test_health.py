"""Phase 2 Milestone 0.2 — health/readiness endpoint tests."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.infrastructure.health.readiness import ReadinessCheck, ReadinessProbe, ReadinessReport


@dataclass(frozen=True)
class _FakeReadinessProbe(ReadinessProbe):
    ready: bool
    message: str = "ok"

    async def check(self) -> ReadinessReport:
        return ReadinessReport(
            ready=self.ready,
            status="ok" if self.ready else "not_ready",
            message=self.message,
            checks=(
                ReadinessCheck(name="configuration", ready=True, message="valid"),
                ReadinessCheck(
                    name="native",
                    ready=self.ready,
                    message=self.message,
                    retryable=not self.ready,
                ),
            ),
        )


def _assert_request_id(response) -> None:
    assert "X-Request-ID" in response.headers
    UUID(response.headers["X-Request-ID"], version=4)


def test_health_live_returns_ok() -> None:
    app = create_app(readiness_probe=_FakeReadinessProbe(ready=True))
    with TestClient(app) as client:
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        _assert_request_id(response)


def test_health_ready_returns_200_when_all_checks_pass() -> None:
    app = create_app(readiness_probe=_FakeReadinessProbe(ready=True))
    with TestClient(app) as client:
        response = client.get("/health/ready")
        assert response.status_code == 200
        _assert_request_id(response)

        data = response.json()
        assert data["status"] == "ok"
        assert data["ready"] is True
        assert "checks" in data
        assert any(c["name"] == "native" and c["ready"] is True for c in data["checks"])


def test_health_ready_returns_503_when_checks_fail() -> None:
    app = create_app(readiness_probe=_FakeReadinessProbe(ready=False, message="gpu_unavailable"))
    with TestClient(app) as client:
        response = client.get("/health/ready")
        assert response.status_code == 503
        _assert_request_id(response)

        data = response.json()
        assert data["status"] == "not_ready"
        assert data["ready"] is False
        assert data["message"] == "gpu_unavailable"
        native_check = next(c for c in data["checks"] if c["name"] == "native")
        assert native_check["ready"] is False
        assert native_check["retryable"] is True


def test_business_endpoints_return_503_when_not_ready() -> None:
    app = create_app(readiness_probe=_FakeReadinessProbe(ready=False))
    with TestClient(app) as client:
        response = client.get("/api/v1/faces/00000000-0000-0000-0000-000000000001")
        assert response.status_code == 503
        body = response.json()
        assert "requestId" in body
        assert body["error"]["code"] == "DEPENDENCY_UNAVAILABLE"

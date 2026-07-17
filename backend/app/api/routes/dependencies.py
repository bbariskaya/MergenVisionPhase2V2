"""FastAPI dependency injection wiring.

Dependencies are exposed as callables so they can be overridden in tests.
"""

from __future__ import annotations

from typing import cast

from fastapi import Request

from app.api.controllers.face_controller import FaceController


def get_face_controller(request: Request) -> FaceController:
    """Return the singleton face controller from app state."""
    return cast(FaceController, request.app.state.face_controller)

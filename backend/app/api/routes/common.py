"""Shared route helpers for thin HTTP endpoints."""

from __future__ import annotations

from typing import NoReturn

from fastapi import HTTPException, Request, status


def raise_not_found(request: Request, code: str, message: str) -> NoReturn:
    """Raise a structured 404 HTTPException."""
    request_id = str(getattr(request.state, "request_id", "unknown"))
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={
            "requestId": request_id,
            "error": {
                "code": code,
                "message": message,
                "retryable": False,
                "details": {},
            },
        },
    )

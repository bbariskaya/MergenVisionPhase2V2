"""Static gate: production worker/runtime path must not re-decode video."""

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[4] / "backend" / "app"
_WORKER_DIRS = [
    _ROOT / "worker",
    _ROOT / "infrastructure" / "runtime",
]

_FORBIDDEN = [
    "ffmpeg",
    "ffprobe",
    "cv2.VideoCapture",
    "VideoCapture",
    "PIL.Image.open",
    "PIL.ImageFile",
]


def _is_forbidden(line: str) -> str | None:
    lower = line.lower()
    for token in _FORBIDDEN:
        if token.lower() in lower:
            return token
    return None


@pytest.mark.parametrize("token", _FORBIDDEN)
def test_no_forbidden_video_decode_in_worker_paths(token: str) -> None:
    violations: list[tuple[Path, int, str]] = []
    for directory in _WORKER_DIRS:
        if not directory.exists():
            continue
        for path in directory.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if token.lower() in line.lower():
                    violations.append((path, lineno, line.strip()))
    if violations:
        formatted = "\n".join(f"  {p}:{n}: {line}" for p, n, line in violations)
        pytest.fail(f"forbidden token {token!r} found:\n{formatted}")

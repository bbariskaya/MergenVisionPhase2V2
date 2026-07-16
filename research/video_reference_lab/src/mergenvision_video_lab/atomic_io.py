"""Atomic file I/O with optional fsync and checksums."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import orjson

from mergenvision_video_lab.hashing import sha256_bytes


class AtomicWriter:
    """Write files atomically via a temporary sibling file + rename."""

    def __init__(self, final_path: Path | str, fsync: bool = True) -> None:
        self.final_path = Path(final_path)
        self.fsync = fsync
        self._temp_path: Path | None = None

    def __enter__(self) -> "AtomicWriter":
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(
            dir=self.final_path.parent,
            prefix=self.final_path.name + ".tmp",
        )
        os.close(fd)
        self._temp_path = Path(name)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            self._cleanup()
            return
        try:
            if self.fsync:
                with open(self._temp_path, "rb") as f:
                    os.fsync(f.fileno())
            os.replace(self._temp_path, self.final_path)
        except Exception:
            self._cleanup()
            raise

    def _cleanup(self) -> None:
        if self._temp_path is not None and self._temp_path.exists():
            try:
                self._temp_path.unlink()
            except OSError:
                pass

    @property
    def temp_path(self) -> Path:
        if self._temp_path is None:
            raise RuntimeError("AtomicWriter not entered")
        return self._temp_path


def write_text_atomic(path: Path | str, text: str) -> None:
    """Write text atomically."""
    with AtomicWriter(path) as writer:
        writer.temp_path.write_text(text, encoding="utf-8")


def write_bytes_atomic(path: Path | str, data: bytes) -> None:
    """Write bytes atomically."""
    with AtomicWriter(path) as writer:
        writer.temp_path.write_bytes(data)


def write_json_atomic(path: Path | str, data: Any) -> None:
    """Write compact JSON atomically."""
    payload = orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY)
    write_bytes_atomic(path, payload)


def write_jsonl_atomic(path: Path | str, records: list[dict[str, Any]]) -> None:
    """Write newline-delimited JSON atomically."""
    lines = [orjson.dumps(r, option=orjson.OPT_SERIALIZE_NUMPY).decode("utf-8") for r in records]
    write_text_atomic(path, "\n".join(lines) + "\n" if lines else "")


def compute_checksums(paths: list[Path]) -> dict[str, str]:
    """Return a dict mapping relative paths to SHA-256 hex digests."""
    result: dict[str, str] = {}
    for p in paths:
        result[p.name] = sha256_bytes(p.read_bytes())
    return result

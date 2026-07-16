"""Unit tests for hashing utilities."""

from __future__ import annotations

from io import BytesIO

from mergenvision_video_lab.hashing import sha256_bytes, sha256_file, sha256_stream


def test_sha256_bytes_known_value(tmp_path) -> None:
    data = b"hello"
    assert sha256_bytes(data) == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_sha256_file(tmp_path) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"hello")
    assert sha256_file(path) == sha256_bytes(b"hello")


def test_sha256_stream() -> None:
    stream = BytesIO(b"hello")
    assert sha256_stream(stream) == sha256_bytes(b"hello")

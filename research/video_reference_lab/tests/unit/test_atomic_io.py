"""Unit tests for atomic I/O."""

from __future__ import annotations

import json

import pytest

from mergenvision_video_lab.atomic_io import (
    AtomicWriter,
    compute_checksums,
    write_bytes_atomic,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)


def test_atomic_writer_creates_final_file(tmp_path) -> None:
    final = tmp_path / "out.txt"
    with AtomicWriter(final) as writer:
        writer.temp_path.write_text("hello", encoding="utf-8")
    assert final.read_text(encoding="utf-8") == "hello"
    assert not any(tmp_path.glob("*.tmp*"))


def test_atomic_writer_removes_temp_on_failure(tmp_path) -> None:
    final = tmp_path / "out.txt"
    with pytest.raises(RuntimeError):
        with AtomicWriter(final) as writer:
            writer.temp_path.write_text("partial", encoding="utf-8")
            raise RuntimeError("abort")
    assert not final.exists()
    assert not any(tmp_path.glob("*.tmp*"))


def test_write_text_atomic(tmp_path) -> None:
    path = tmp_path / "a.txt"
    write_text_atomic(path, "world")
    assert path.read_text(encoding="utf-8") == "world"


def test_write_bytes_atomic(tmp_path) -> None:
    path = tmp_path / "b.bin"
    write_bytes_atomic(path, b"\x00\x01\x02")
    assert path.read_bytes() == b"\x00\x01\x02"


def test_write_json_atomic(tmp_path) -> None:
    path = tmp_path / "c.json"
    write_json_atomic(path, {"x": [1, 2, 3]})
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == {"x": [1, 2, 3]}


def test_write_jsonl_atomic(tmp_path) -> None:
    path = tmp_path / "d.jsonl"
    write_jsonl_atomic(path, [{"a": 1}, {"b": 2}])
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"a": 1}


def test_compute_checksums(tmp_path) -> None:
    (tmp_path / "one.txt").write_text("one")
    (tmp_path / "two.txt").write_text("two")
    checksums = compute_checksums([tmp_path / "one.txt", tmp_path / "two.txt"])
    assert "one.txt" in checksums
    assert "two.txt" in checksums
    assert len(set(checksums.values())) == 2

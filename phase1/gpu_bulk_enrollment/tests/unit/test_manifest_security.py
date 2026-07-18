"""Tests for manifest validation and path-traversal defences."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from mv_phase1_bulk.manifest import EnrollmentManifest, ManifestValidationError


def _write_manifest(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return manifest


def test_basic_manifest_loads(tmp_path: Path) -> None:
    img = tmp_path / "a.jpg"
    img.write_bytes(b"jpeg")
    manifest = _write_manifest(
        tmp_path,
        [{"subject_key": "alice", "image_paths": ["a.jpg"]}],
    )
    enrollment = EnrollmentManifest.from_file(tmp_path, manifest)
    assert len(enrollment) == 1
    assert enrollment.total_images() == 1
    assert enrollment.records[0].subject_key == "alice"


def test_empty_manifest_rejected(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, [])
    with pytest.raises(ManifestValidationError, match="no subjects"):
        EnrollmentManifest.from_file(tmp_path, manifest)


def test_missing_subject_key_rejected(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, [{"image_paths": ["a.jpg"]}])
    with pytest.raises(ManifestValidationError, match="subject_key"):
        EnrollmentManifest.from_file(tmp_path, manifest)


def test_missing_image_paths_rejected(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, [{"subject_key": "alice"}])
    with pytest.raises(ManifestValidationError, match="image_paths"):
        EnrollmentManifest.from_file(tmp_path, manifest)


def test_missing_image_file_rejected(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        [{"subject_key": "alice", "image_paths": ["missing.jpg"]}],
    )
    with pytest.raises(ManifestValidationError, match="not found"):
        EnrollmentManifest.from_file(tmp_path, manifest)


def test_absolute_image_path_rejected(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        [{"subject_key": "alice", "image_paths": ["/etc/passwd"]}],
    )
    with pytest.raises(ManifestValidationError, match="absolute"):
        EnrollmentManifest.from_file(tmp_path, manifest)


def test_path_traversal_rejected(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        [{"subject_key": "alice", "image_paths": ["../secret.jpg"]}],
    )
    with pytest.raises(ManifestValidationError, match="traversal"):
        EnrollmentManifest.from_file(tmp_path, manifest)


def test_sha256_mismatch_rejected(tmp_path: Path) -> None:
    img = tmp_path / "a.jpg"
    img.write_bytes(b"jpeg")
    manifest = _write_manifest(
        tmp_path,
        [{"subject_key": "alice", "image_paths": ["a.jpg"], "sha256": "0" * 64}],
    )
    with pytest.raises(ManifestValidationError, match="SHA256 mismatch"):
        EnrollmentManifest.from_file(tmp_path, manifest, require_sha256=True)


def test_sha256_match_accepted(tmp_path: Path) -> None:
    img = tmp_path / "a.jpg"
    img.write_bytes(b"jpeg")
    digest = hashlib.sha256(b"jpeg").hexdigest()
    manifest = _write_manifest(
        tmp_path,
        [{"subject_key": "alice", "image_paths": ["a.jpg"], "sha256": digest}],
    )
    enrollment = EnrollmentManifest.from_file(tmp_path, manifest, require_sha256=True)
    assert enrollment.total_images() == 1


def test_disabled_path_validation_loads(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        [{"subject_key": "alice", "image_paths": ["missing.jpg"]}],
    )
    enrollment = EnrollmentManifest.from_file(tmp_path, manifest, validate_paths=False)
    assert enrollment.total_images() == 1

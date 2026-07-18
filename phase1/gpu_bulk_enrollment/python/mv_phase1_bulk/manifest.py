"""Dataset manifest validation and loading.

Manifest format (JSONL, one record per line):

    {"subject_key": "person_a", "image_paths": ["a/001.jpg", "a/002.jpg"]}
    {"subject_key": "person_b", "image_paths": ["b/001.jpg"]}

`subject_key` is the stable external identifier for a person.
`image_paths` are relative to the dataset root.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class ManifestValidationError(Exception):
    """Raised when the manifest or dataset fails validation."""


class SubjectRecord:
    """One subject with validated image paths."""

    def __init__(
        self,
        subject_key: str,
        image_paths: list[Path],
    ) -> None:
        self.subject_key = subject_key
        self.image_paths = image_paths


class EnrollmentManifest:
    """Validated enrollment manifest with relative dataset paths."""

    def __init__(
        self,
        dataset_root: Path,
        records: list[SubjectRecord],
        max_file_size_bytes: int = 50 * 1024 * 1024,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.records = records
        self.max_file_size_bytes = max_file_size_bytes

    @classmethod
    def from_file(
        cls,
        dataset_root: Path,
        manifest_path: Path,
        *,
        max_file_size_bytes: int = 50 * 1024 * 1024,
        validate_paths: bool = True,
        require_sha256: bool = False,
    ) -> EnrollmentManifest:
        dataset_root = Path(dataset_root).resolve()
        manifest_path = Path(manifest_path).resolve()
        if not dataset_root.is_dir():
            raise ManifestValidationError(f"dataset root is not a directory: {dataset_root}")
        if not manifest_path.exists():
            raise ManifestValidationError(f"manifest not found: {manifest_path}")

        records: list[SubjectRecord] = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ManifestValidationError(
                        f"manifest line {line_no}: invalid JSON: {exc}"
                    ) from exc
                record = cls._parse_record(
                    obj,
                    dataset_root,
                    line_no,
                    validate_paths=validate_paths,
                    require_sha256=require_sha256,
                    max_file_size_bytes=max_file_size_bytes,
                )
                records.append(record)

        if not records:
            raise ManifestValidationError("manifest contains no subjects")

        return cls(dataset_root, records, max_file_size_bytes=max_file_size_bytes)

    @classmethod
    def _parse_record(
        cls,
        obj: dict[str, Any],
        dataset_root: Path,
        line_no: int,
        *,
        validate_paths: bool,
        require_sha256: bool,
        max_file_size_bytes: int,
    ) -> SubjectRecord:
        subject_key = obj.get("subject_key")
        if not isinstance(subject_key, str) or not subject_key:
            raise ManifestValidationError(
                f"manifest line {line_no}: missing or invalid subject_key"
            )

        image_paths_raw = obj.get("image_paths")
        if not isinstance(image_paths_raw, list) or not image_paths_raw:
            raise ManifestValidationError(
                f"manifest line {line_no}: missing or empty image_paths"
            )

        image_paths: list[Path] = []
        for p in image_paths_raw:
            if not isinstance(p, str):
                raise ManifestValidationError(
                    f"manifest line {line_no}: image_paths must be strings"
                )
            rel = Path(p)
            if rel.is_absolute():
                raise ManifestValidationError(
                    f"manifest line {line_no}: absolute image path not allowed: {p}"
                )
            resolved = (dataset_root / rel).resolve()
            # Path traversal guard: resolved must be under dataset_root.
            try:
                resolved.relative_to(dataset_root)
            except ValueError as exc:
                raise ManifestValidationError(
                    f"manifest line {line_no}: path traversal detected: {p}"
                ) from exc

            if validate_paths:
                if not resolved.exists():
                    raise ManifestValidationError(
                        f"manifest line {line_no}: image not found: {resolved}"
                    )
                stat = resolved.stat()
                if stat.st_size > max_file_size_bytes:
                    raise ManifestValidationError(
                        f"manifest line {line_no}: image exceeds size limit: {resolved}"
                    )
                if require_sha256:
                    expected = obj.get("sha256")
                    if expected:
                        actual = cls._sha256_file(resolved)
                        if actual != expected:
                            raise ManifestValidationError(
                                f"manifest line {line_no}: SHA256 mismatch for {resolved}"
                            )

            image_paths.append(resolved)

        if not image_paths:
            raise ManifestValidationError(
                f"manifest line {line_no}: no valid image paths for subject {subject_key}"
            )

        return SubjectRecord(subject_key=subject_key, image_paths=image_paths)

    @staticmethod
    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[SubjectRecord]:
        return iter(self.records)

    def total_images(self) -> int:
        return sum(len(r.image_paths) for r in self.records)

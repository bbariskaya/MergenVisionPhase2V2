"""Content-addressed frozen artifact storage with resume and checksums."""

from __future__ import annotations

import contextlib
import fcntl
import json
from pathlib import Path
from typing import Any

import numpy as np

from mergenvision_video_lab.atomic_io import (
    write_bytes_atomic,
    write_json_atomic,
    write_jsonl_atomic,
)
from mergenvision_video_lab.contracts import FaceObservation, FrameRecord, RunManifest
from mergenvision_video_lab.errors import ArtifactCorruptError
from mergenvision_video_lab.hashing import sha256_file


class RunLock:
    """Advisory file lock to prevent concurrent writes to a run directory."""

    def __init__(self, run_dir: Path) -> None:
        self.lock_path = run_dir / ".run.lock"
        self._file: Any = None

    def __enter__(self) -> RunLock:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.lock_path.open("w")
        fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._file is not None:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            with contextlib.suppress(Exception):
                self._file.close()


class ArtifactStore:
    """Read/write frozen observations, embeddings, and manifest atomically."""

    SCHEMA_VERSIONS = {"mv-video-reference-manifest/v1", "mv-face-observation/v1"}

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.raw_dir = self.run_dir / "raw"
        self.manifest_path = self.raw_dir / "manifest.json"
        self.frames_path = self.raw_dir / "frames.jsonl"
        self.observations_path = self.raw_dir / "observations.jsonl"
        self.embeddings_path = self.raw_dir / "embeddings.npy"
        self.checksums_path = self.raw_dir / "checksums.sha256"

    def is_complete(self) -> bool:
        """Return True if all expected raw artifacts exist."""
        return (
            self.manifest_path.exists()
            and self.frames_path.exists()
            and self.observations_path.exists()
            and self.embeddings_path.exists()
            and self.checksums_path.exists()
        )

    def _paths_for_checksum(self) -> list[Path]:
        return [
            self.manifest_path,
            self.frames_path,
            self.observations_path,
            self.embeddings_path,
        ]

    def write_manifest(self, manifest: RunManifest) -> None:
        write_json_atomic(self.manifest_path, manifest.model_dump(mode="json"))

    def write_frames(self, frames: list[FrameRecord]) -> None:
        records = [frame.model_dump(mode="json") for frame in frames]
        write_jsonl_atomic(self.frames_path, records)

    def write_observations(self, observations: list[FaceObservation]) -> None:
        records = [obs.model_dump(mode="json") for obs in observations]
        write_jsonl_atomic(self.observations_path, records)

    def write_embeddings(self, embeddings: np.ndarray) -> None:
        if embeddings.dtype != np.float32:
            raise ArtifactCorruptError("embeddings must be float32")
        if embeddings.ndim != 2 or embeddings.shape[1] != 512:
            raise ArtifactCorruptError("embeddings must have shape [N, 512]")
        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.embeddings_path.with_suffix(".tmp.npy")
        np.save(tmp_path, embeddings)
        tmp_path.replace(self.embeddings_path)

    def write_checksums(self) -> None:
        lines = []
        for p in self._paths_for_checksum():
            digest = sha256_file(p)
            lines.append(f"{digest}  {p.name}\n")
        write_bytes_atomic(self.checksums_path, "".join(lines).encode("utf-8"))

    def read_manifest(self) -> RunManifest:
        with open(self.manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        return RunManifest(**data)

    def read_observations(self) -> list[FaceObservation]:
        observations: list[FaceObservation] = []
        if not self.observations_path.exists():
            return observations
        with open(self.observations_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                observations.append(FaceObservation(**json.loads(line)))
        return observations

    def read_frames(self) -> list[FrameRecord]:
        frames: list[FrameRecord] = []
        if not self.frames_path.exists():
            return frames
        with open(self.frames_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                frames.append(FrameRecord(**json.loads(line)))
        return frames

    def read_embeddings(self) -> np.ndarray:
        return np.asarray(np.load(self.embeddings_path), dtype=np.float32)

    def validate(self) -> None:
        """Validate resume preconditions.

        Raises ArtifactCorruptError if anything is inconsistent.
        """
        if not self.is_complete():
            raise ArtifactCorruptError("run directory is incomplete")

        manifest = self.read_manifest()
        if manifest.schema_version != "mv-video-reference-manifest/v1":
            raise ArtifactCorruptError(
                f"unexpected manifest schema version: {manifest.schema_version}"
            )

        frames = self.read_frames()
        if len(frames) != manifest.decoded_frame_count:
            raise ArtifactCorruptError(
                f"frame count mismatch: {len(frames)} vs {manifest.decoded_frame_count}"
            )
        sampled_count = sum(1 for f in frames if f.sampled)
        if sampled_count != manifest.sampled_frame_count:
            raise ArtifactCorruptError(
                f"sampled frame count mismatch: {sampled_count} vs {manifest.sampled_frame_count}"
            )
        processed_count = sum(1 for f in frames if f.processed)
        if processed_count != manifest.processed_frame_count:
            raise ArtifactCorruptError(
                f"processed frame count mismatch: {processed_count} vs {manifest.processed_frame_count}"
            )
        # Frame indices must be unique and monotonically increasing starting at 0.
        expected_indices = list(range(len(frames)))
        actual_indices = [f.frame_index for f in frames]
        if actual_indices != expected_indices:
            raise ArtifactCorruptError(
                "frame indices are not contiguous from 0",
                {
                    "expected": expected_indices[:5] + ["..."],
                    "actual": actual_indices[:5] + ["..."],
                },
            )

        observations = self.read_observations()
        if len(observations) != manifest.observation_count:
            raise ArtifactCorruptError(
                f"observation count mismatch: {len(observations)} vs {manifest.observation_count}"
            )

        embeddings = self.read_embeddings()
        if embeddings.shape[0] != manifest.valid_embedding_count:
            raise ArtifactCorruptError(
                f"embedding count mismatch: {embeddings.shape[0]} vs {manifest.valid_embedding_count}"
            )
        if embeddings.ndim != 2 or embeddings.shape[1] != 512:
            raise ArtifactCorruptError(f"unexpected embedding shape: {embeddings.shape}")
        if not np.all(np.isfinite(embeddings)):
            raise ArtifactCorruptError("embeddings contain non-finite values")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ArtifactCorruptError("embeddings are not unit-normalized")

        valid_indices = {
            obs.embedding_index for obs in observations if obs.embedding_index is not None
        }
        if valid_indices and (max(valid_indices) >= embeddings.shape[0] or min(valid_indices) < 0):
            raise ArtifactCorruptError("observation embedding_index out of bounds")

        # Checksum validation.
        if self.checksums_path.exists():
            with open(self.checksums_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    expected, name = line.split(maxsplit=1)
                    actual = sha256_file(self.raw_dir / name)
                    if actual != expected:
                        raise ArtifactCorruptError(
                            f"checksum mismatch for {name}",
                            {"expected": expected, "actual": actual},
                        )

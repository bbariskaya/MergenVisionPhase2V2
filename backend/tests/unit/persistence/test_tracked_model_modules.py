"""Fresh-checkout contract: every production ORM module is tracked and importable."""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import pytest

from app.infrastructure.persistence.sqlalchemy.base import Base

REQUIRED_PHASE2_TABLES = {
    "appearance_interval",
    "idempotency_record",
    "outbox_event",
    "process_event",
    "video_asset",
    "video_job",
    "video_timeline_chunk",
    "video_track",
    "video_track_sample",
    "video_tracklet",
}

REQUIRED_MODULES = [
    "app.infrastructure.persistence.sqlalchemy.models.appearance_interval",
    "app.infrastructure.persistence.sqlalchemy.models.idempotency_record",
    "app.infrastructure.persistence.sqlalchemy.models.outbox_event",
    "app.infrastructure.persistence.sqlalchemy.models.process_event",
    "app.infrastructure.persistence.sqlalchemy.models.video_asset",
    "app.infrastructure.persistence.sqlalchemy.models.video_job",
    "app.infrastructure.persistence.sqlalchemy.models.video_timeline_chunk",
    "app.infrastructure.persistence.sqlalchemy.models.video_track",
    "app.infrastructure.persistence.sqlalchemy.models.video_track_sample",
    "app.infrastructure.persistence.sqlalchemy.models.video_tracklet",
]


def _git_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def _module_to_path(module_name: str, root: Path) -> Path:
    relative = module_name.replace(".", "/") + ".py"
    return root / "backend" / relative


def _is_ignored_by_gitignore(path: Path, root: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(path)],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        pytest.fail("git binary not available; cannot confirm .gitignore safety")
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    # 128 or other -> fail closed
    pytest.fail(f"git check-ignore failed for {path}: {result.stderr}")


@pytest.mark.parametrize("module_name", REQUIRED_MODULES)
def test_required_orm_module_file_exists_and_not_ignored(module_name: str) -> None:
    root = _git_root()
    path = _module_to_path(module_name, root)
    assert path.exists(), f"required ORM module missing: {path}"
    assert path.is_file(), f"required ORM path is not a file: {path}"
    assert not _is_ignored_by_gitignore(path, root), f"required ORM module ignored by .gitignore: {path}"


@pytest.mark.parametrize("module_name", REQUIRED_MODULES)
def test_required_orm_module_importable(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module is not None
    # Ensure every required module registers at least one table on the shared Base.
    assert any(table.startswith(module_name.rsplit(".", 1)[-1]) for table in Base.metadata.tables), (
        f"{module_name} did not register a table on Base.metadata"
    )


def test_all_phase2_tables_registered_in_metadata() -> None:
    tables = set(Base.metadata.tables.keys())
    missing = REQUIRED_PHASE2_TABLES - tables
    assert not missing, f"Phase 2 tables missing from Base.metadata: {missing}"

"""Ensure layer boundaries remain clean."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).parents[2]
APP_DIR = BACKEND_DIR / "app"

FORBIDDEN_INFRASTRUCTURE = {
    "sqlalchemy",
    "asyncpg",
    "minio",
    "qdrant_client",
    "fastapi",
}


def _collect_python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    if not root.exists():
        return files
    for path in root.rglob("*.py"):
        if path.name == "__init__.py" and not path.read_text().strip():
            continue
        files.append(path)
    return files


def _imports_in_file(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                found.add(root)
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            found.add(root)
            # Capture intra-project infrastructure imports as well.
            if node.module.startswith("app.infrastructure"):
                found.add("app.infrastructure")
    return found


@pytest.mark.parametrize(
    "source_file",
    _collect_python_files(APP_DIR / "domain"),
    ids=lambda p: str(p.relative_to(APP_DIR.parents[1])),
)
def test_domain_file_does_not_import_infrastructure(source_file: Path) -> None:
    imports = _imports_in_file(source_file)
    violations = imports & FORBIDDEN_INFRASTRUCTURE
    assert not violations, f"{source_file} imports {violations}"


@pytest.mark.parametrize(
    "source_file",
    _collect_python_files(APP_DIR / "application"),
    ids=lambda p: str(p.relative_to(APP_DIR.parents[1])),
)
def test_application_file_does_not_import_infrastructure(source_file: Path) -> None:
    imports = _imports_in_file(source_file)
    violations = imports & FORBIDDEN_INFRASTRUCTURE
    assert not violations, f"{source_file} imports {violations}"
    assert "app.infrastructure" not in imports, f"{source_file} imports app.infrastructure"

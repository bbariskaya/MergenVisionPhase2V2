"""Ensure the domain layer remains a pure-Python dependency-free core."""

import ast
from pathlib import Path

import pytest

DOMAIN_DIR = Path(__file__).parents[2] / "app" / "domain"
FORBIDDEN_TOP_LEVEL = {"sqlalchemy", "asyncpg", "minio", "qdrant_client", "fastapi"}


def _collect_domain_python_files() -> list[Path]:
    files: list[Path] = []
    if not DOMAIN_DIR.exists():
        return files
    for path in DOMAIN_DIR.rglob("*.py"):
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
    return found


@pytest.mark.parametrize("domain_file", _collect_domain_python_files())
def test_domain_file_does_not_import_infrastructure(domain_file: Path) -> None:
    imports = _imports_in_file(domain_file)
    violations = imports & FORBIDDEN_TOP_LEVEL
    assert not violations, f"{domain_file.relative_to(DOMAIN_DIR.parents[1])} imports {violations}"

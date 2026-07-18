"""Phase 2 protection gate logic and git helpers.

Only two prefixes may change after the task-start baseline:

  - phase1/gpu_bulk_enrollment/
  - .artifacts/phase1_gpu_bulk_enrollment/

Everything else is protected. Paths that were already dirty at task start are
allowed to remain dirty but may not be modified.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, cast

ALLOWLIST_PREFIXES = (
    "phase1/gpu_bulk_enrollment/",
    ".artifacts/phase1_gpu_bulk_enrollment/",
)

VIOLATION = 1
MISSING_BASELINE = 2
HEAD_ADVANCED = 3


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def run_ledger_dir(repo_root: Path) -> Path:
    return repo_root / ".artifacts" / "phase1_gpu_bulk_enrollment" / "runs"


def latest_baseline(repo_root: Path) -> dict[str, Any]:
    ledger = run_ledger_dir(repo_root)
    if not ledger.exists():
        raise FileNotFoundError("no run ledger directory found")
    runs = sorted(ledger.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for run in runs:
        baseline = run / "baseline.json"
        if baseline.exists():
            return cast(dict[str, Any], json.loads(baseline.read_text(encoding="utf-8")))
    raise FileNotFoundError("no run ledger baseline found")


def git_status_paths(repo_root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "status", "--short", "-uall"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    paths: set[str] = set()
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        rest = line[3:]
        if " -> " in rest:
            rest = rest.split(" -> ")[-1]
        paths.add(rest.strip())
    return paths


def git_diff_paths(repo_root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return {p for p in result.stdout.splitlines() if p.strip()}


def git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def is_allowlisted(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in ALLOWLIST_PREFIXES)


def classify(
    current_paths: set[str],
    baseline_untracked: set[str],
    current_head: str,
    baseline_head: str,
) -> tuple[set[str], set[str], str | None]:
    """Return (violations, baseline_dirty, head_error).

    ``baseline_dirty`` are paths outside the allowlist that were already dirty
    at baseline and remain present in the working tree.
    """
    if current_head != baseline_head:
        return set(), set(), f"HEAD advanced: expected {baseline_head}, got {current_head}"

    violations: set[str] = set()
    baseline_dirty: set[str] = set()
    for path in current_paths:
        if is_allowlisted(path):
            continue
        if path in baseline_untracked:
            baseline_dirty.add(path)
            continue
        violations.add(path)
    return violations, baseline_dirty, None

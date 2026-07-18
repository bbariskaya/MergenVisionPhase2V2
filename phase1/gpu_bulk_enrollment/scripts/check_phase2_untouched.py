#!/usr/bin/env python3
"""Phase 2 protection gate entrypoint."""
from __future__ import annotations

import sys

from mv_phase1_bulk._untouched_gate import (
    MISSING_BASELINE,
    VIOLATION,
    HEAD_ADVANCED,
    classify,
    git_diff_paths,
    git_head,
    git_status_paths,
    latest_baseline,
    repo_root,
)


def main() -> int:
    root = repo_root()
    try:
        baseline = latest_baseline(root)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return MISSING_BASELINE

    baseline_head = baseline["head_sha"]
    baseline_untracked = set(baseline.get("baseline_untracked", []))
    current_paths = git_status_paths(root) | git_diff_paths(root)

    violations, baseline_dirty, head_error = classify(
        current_paths,
        baseline_untracked,
        git_head(root),
        baseline_head,
    )

    if head_error:
        print(f"FAIL: {head_error}", file=sys.stderr)
        return HEAD_ADVANCED

    for path in sorted(baseline_dirty):
        print(f"BASELINE_DIRTY: {path}", file=sys.stderr)

    if violations:
        for path in sorted(violations):
            print(f"ERROR: protected path changed: {path}", file=sys.stderr)
        print(f"FAIL: {len(violations)} Phase 2 protection violation(s)", file=sys.stderr)
        return VIOLATION

    print("PASS: Phase 2 tree untouched; all changes are under allowlist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

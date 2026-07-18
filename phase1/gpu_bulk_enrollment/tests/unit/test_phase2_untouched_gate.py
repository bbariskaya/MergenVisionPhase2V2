"""Unit tests for the Phase 2 protection gate classification logic."""

from __future__ import annotations

from mv_phase1_bulk._untouched_gate import classify, is_allowlisted


def test_allowlisted_paths_are_ignored() -> None:
    violations, dirty, err = classify(
        {
            "phase1/gpu_bulk_enrollment/python/foo.py",
            ".artifacts/phase1_gpu_bulk_enrollment/runs/abc/baseline.json",
        },
        set(),
        "abc123",
        "abc123",
    )
    assert not violations
    assert not dirty
    assert err is None


def test_baseline_untracked_allowed() -> None:
    violations, dirty, err = classify(
        {"prompt14.txt", ".claude/projectsummary.md"},
        {"prompt14.txt", ".claude/projectsummary.md"},
        "abc123",
        "abc123",
    )
    assert not violations
    assert dirty == {"prompt14.txt", ".claude/projectsummary.md"}
    assert err is None


def test_new_untracked_outside_allowlist_is_violation() -> None:
    violations, dirty, err = classify(
        {"backend/app/main.py", "frontend/src/App.tsx"},
        set(),
        "abc123",
        "abc123",
    )
    assert violations == {"backend/app/main.py", "frontend/src/App.tsx"}
    assert not dirty
    assert err is None


def test_head_advanced_is_error() -> None:
    violations, dirty, err = classify(set(), set(), "newhead", "oldhead")
    assert not violations
    assert not dirty
    assert err is not None
    assert "HEAD advanced" in err


def test_is_allowlisted() -> None:
    assert is_allowlisted("phase1/gpu_bulk_enrollment/python/foo.py")
    assert is_allowlisted(".artifacts/phase1_gpu_bulk_enrollment/engines/x.engine")
    assert not is_allowlisted("backend/app/main.py")
    assert not is_allowlisted("prompt14.txt")

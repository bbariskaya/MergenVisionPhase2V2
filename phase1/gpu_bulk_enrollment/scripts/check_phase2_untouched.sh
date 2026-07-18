#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASELINE_DIR="${REPO_ROOT}/.artifacts/phase1_gpu_bulk_enrollment/baseline"
MANIFEST="${BASELINE_DIR}/protected_tree_manifest.txt"
ALLOWLIST_PREFIX="phase1/gpu_bulk_enrollment/"
RUNTIME_ARTIFACT_PREFIX=".artifacts/phase1_gpu_bulk_enrollment/"

if [[ ! -f "${MANIFEST}" ]]; then
    echo "ERROR: baseline manifest not found: ${MANIFEST}" >&2
    echo "Run baseline capture first (Adim 1)." >&2
    exit 2
fi

cd "${REPO_ROOT}"

# Verify tracked protected tree has not changed since baseline.
CURRENT_MANIFEST="$(mktemp)"
trap 'rm -f "${CURRENT_MANIFEST}"' EXIT
git ls-tree -r HEAD > "${CURRENT_MANIFEST}"

if ! diff -q "${MANIFEST}" "${CURRENT_MANIFEST}" >/dev/null 2>&1; then
    echo "ERROR: protected tracked tree changed since baseline" >&2
    diff "${MANIFEST}" "${CURRENT_MANIFEST}" >&2 || true
    exit 1
fi

# Any source changes in the working tree must be under the allowlist prefix.
# Staged/untracked changes are both checked.
VIOLATIONS=0

# Pre-existing dirty files known to be user-owned at baseline are not our fault,
# but we still flag them so the final report is honest.
USER_DIRTY_FILES=(
    "prompt13.txt"
    ".claude/plans/phase1-gpu-bulk-enrollment-plan.md"
)

is_user_dirty() {
    local path="$1"
    for known in "${USER_DIRTY_FILES[@]}"; do
        if [[ "${path}" == "${known}" ]]; then
            return 0
        fi
    done
    return 1
}

# Helper: check if path is protected (i.e. not in allowlist/runtime prefix).
is_protected_change() {
    local path="$1"
    if [[ "${path}" == "${ALLOWLIST_PREFIX}"* ]]; then
        return 1
    fi
    if [[ "${path}" == "${RUNTIME_ARTIFACT_PREFIX}"* ]]; then
        return 1
    fi
    return 0
}

# Check git diff --name-only (modified/staged tracked files).
while IFS= read -r path; do
    [[ -z "${path}" ]] && continue
    if is_protected_change "${path}"; then
        if is_user_dirty "${path}"; then
            echo "WARN: pre-existing user dirty file (not counted as violation): ${path}" >&2
        else
            echo "ERROR: protected file changed: ${path}" >&2
            VIOLATIONS=$((VIOLATIONS + 1))
        fi
    fi
done < <(git diff --name-only)

# Check git status --short -uall for untracked files.
while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    # Status short format: XY path or XY path -> path for renames.
    path="${line:3}"
    # Handle rename arrow.
    if [[ "${path}" == *" -> "* ]]; then
        path="${path##* -> }"
    fi
    if is_protected_change "${path}"; then
        if is_user_dirty "${path}"; then
            echo "WARN: pre-existing user untracked file (not counted as violation): ${path}" >&2
        else
            echo "ERROR: protected path has new/untracked change: ${path}" >&2
            VIOLATIONS=$((VIOLATIONS + 1))
        fi
    fi
done < <(git status --short -uall)

if [[ ${VIOLATIONS} -gt 0 ]]; then
    echo "FAIL: ${VIOLATIONS} Phase 2 protection violation(s) detected." >&2
    exit 1
fi

echo "PASS: Phase 2 tree untouched; all working-tree changes are under ${ALLOWLIST_PREFIX}."

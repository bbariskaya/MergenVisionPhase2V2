# Phase 1 Isolated Native GPU Bulk Enrollment

Isolated, read-only-dataset → native GPU batch enrollment tool for MergenVisionPhase2v2.

## Scope

- All source lives under `phase1/gpu_bulk_enrollment/`.
- Runtime artifacts live under `.artifacts/phase1_gpu_bulk_enrollment/` or package-local `artifacts/`.
- Phase 2 source (`backend/**`, `frontend/**`, `research/**`, root Makefile, Docker, etc.) is **untouched**.
- Uses MergenVisionDemo's proven native GPU batch architecture as reference.

## Layout

```
phase1/gpu_bulk_enrollment/
  README.md                 # this file
  pyproject.toml            # Python package + scikit-build-core native build
  CMakeLists.txt            # native CUDA/pybind11 module
  Makefile                  # developer targets
  Dockerfile.gpu            # pinned GPU runtime/build image
  docker-compose.gpu.yml    # local orchestration
  config/                   # model profile + enrollment + benchmark schemas
  native/                   # C++/CUDA source
  python/mv_phase1_bulk/    # Python control plane
  scripts/                  # build, gate, fixture, benchmark scripts
  tests/                    # unit, contract, integration, gpu, acceptance
  docs/                     # reference source map, target contract, review package
  fixtures/                 # manifest fixtures (no real images in git)
  artifacts/                # gitignored runtime outputs
```

## Quick Start

```bash
make -C phase1/gpu_bulk_enrollment static     # lint/type/import checks
make -C phase1/gpu_bulk_enrollment build      # native build + wheel
make -C phase1/gpu_bulk_enrollment unit       # unit tests
make -C phase1/gpu_bulk_enrollment contract   # native contract tests
make -C phase1/gpu_bulk_enrollment gpu-smoke  # real GPU smoke
make -C phase1/gpu_bulk_enrollment e2e        # real PG/MinIO/Qdrant E2E
make -C phase1/gpu_bulk_enrollment benchmark  # benchmark matrix
make -C phase1/gpu_bulk_enrollment acceptance # full acceptance + phase2 continuity
make -C phase1/gpu_bulk_enrollment phase2-untouched-gate
```

## CLI

```bash
mv-phase1-bulk inspect-models --profile config/model_profile.json
mv-phase1-bulk build-engines --profile config/model_profile.json
mv-phase1-bulk validate-manifest --dataset-root /dataset --manifest /dataset/manifest.jsonl
mv-phase1-bulk enroll --dataset-root /dataset --manifest /dataset/manifest.jsonl \
  --workers 1 --gpu-devices 0 --batch-size 16 --resume
mv-phase1-bulk benchmark --dataset-root /dataset --manifest /dataset/benchmark.jsonl \
  --gpu-devices 0,1,2 --batch-matrix 1,2,4,8,16 --runs 3
mv-phase1-bulk reconcile --run-id <run-id>
mv-phase1-bulk report --run-id <run-id>
```

## Protection

The `scripts/check_phase2_untouched.sh` gate verifies that no Phase 2 source file has changed.
Run it before and after every significant step.

.PHONY: static unit contract build gpu-smoke integration e2e benchmark retry-acceptance phase2-continuity acceptance phase2-untouched-gate clean

ROOT := $(abspath $(dir $(MAKEFILE_LIST))/../..)
VENV := $(ROOT)/.artifacts/phase1_gpu_bulk_enrollment/venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PACKAGE := mv_phase1_bulk
export PYTHONPATH := python

static:
	$(PYTHON) -m ruff check python/ tests/
	$(PYTHON) -m ruff format --check python/ tests/
	$(PYTHON) -m mypy python/
	$(PYTHON) -c "import $(PACKAGE); print($(PACKAGE).__version__)"

unit:
	$(PYTHON) -m pytest tests/unit -v

contract:
	$(PYTHON) -m pytest tests/contract -v

build:
	$(PIP) install -e . --no-build-isolation -v

gpu-smoke:
	$(PYTHON) -m pytest tests/gpu -v -m gpu

integration:
	$(PYTHON) -m pytest tests/integration -v

e2e:
	$(PYTHON) -m pytest tests/acceptance/test_e2e.py -v

benchmark:
	$(PYTHON) -m $(PACKAGE).benchmark --dataset-root /dataset --manifest /dataset/benchmark.jsonl --gpu-devices 0 --batch-matrix 1,2,4,8,16 --runs 3

retry-acceptance:
	$(PYTHON) -m pytest tests/acceptance/test_retry.py -v

phase2-continuity:
	$(PYTHON) -m pytest tests/acceptance/test_phase2_continuity.py -v

acceptance: static unit contract build gpu-smoke integration e2e benchmark retry-acceptance phase2-continuity phase2-untouched-gate
	@echo "ACCEPTANCE PASS"

phase2-untouched-gate:
	./scripts/check_phase2_untouched.sh

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.so" -delete

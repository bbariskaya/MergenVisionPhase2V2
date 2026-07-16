# MergenVision Phase 1 Sprint 01 — Minimal Identity Storage Foundation
#
# All acceptance targets operate against the dedicated test namespace defined
# in docker-compose.test.yml and backend/.env.test. They never touch the
# development services in docker-compose.yml or any production resources.

COMPOSE_PROJECT := mergenvision-s01-test
COMPOSE_FILE    := docker-compose.test.yml
TEST_COMPOSE    := docker compose -p $(COMPOSE_PROJECT) -f $(COMPOSE_FILE)
TEST_SERVICES   := postgres-test minio-test qdrant-test

BACKEND_DIR     := backend
PYTHON          := $(PWD)/$(BACKEND_DIR)/.venv/bin/python
PYTEST          := $(PYTHON) -m pytest
RUFF            := $(PYTHON) -m ruff
MYPY            := $(PYTHON) -m mypy
ALEMBIC         := $(PYTHON) -m alembic

WITH_TEST_ENV   := set -a && . $(PWD)/$(BACKEND_DIR)/.env.test &&
TEST_REPORTS    := $(PWD)/test-reports

.PHONY: bootstrap \
        phase1-sprint-01-preflight \
        phase1-sprint-01-services \
        phase1-sprint-01-static phase1-sprint-01-format-check \
        phase1-sprint-01-unit phase1-sprint-01-integration \
        phase1-sprint-01-full-test phase1-sprint-01-restart \
        phase1-sprint-01-image-build \
        phase1-sprint-01-down phase1-sprint-01-acceptance

# ------------------------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------------------------

bootstrap:
	@echo "==> Creating local environment files (only if missing)..."
	test -f $(BACKEND_DIR)/.env.test || cp $(BACKEND_DIR)/.env.test.example $(BACKEND_DIR)/.env.test
	test -f $(BACKEND_DIR)/.env || cp $(BACKEND_DIR)/.env.example $(BACKEND_DIR)/.env
	@echo "==> Creating Python virtual environment..."
	python3.12 -m venv $(BACKEND_DIR)/.venv
	@echo "==> Installing Python dependencies..."
	cd $(BACKEND_DIR) && $(PYTHON) -m pip install --upgrade pip
	cd $(BACKEND_DIR) && $(PYTHON) -m pip install -r requirements.lock
	cd $(BACKEND_DIR) && $(PYTHON) -m pip check
	@echo "==> Pulling test service images..."
	$(TEST_COMPOSE) pull
	@echo "==> Bootstrap complete. Run 'make phase1-sprint-01-acceptance' to verify."

# ------------------------------------------------------------------------------
# Preflight / safety
# ------------------------------------------------------------------------------

phase1-sprint-01-preflight:
	@echo "==> Preflight: fail-closed guard unit tests..."
	cd $(BACKEND_DIR) && $(WITH_TEST_ENV) $(PYTEST) tests/unit/support/test_resource_guard.py -v
	@echo "==> Preflight: Docker Compose config validation..."
	$(TEST_COMPOSE) config > /dev/null
	docker compose -f docker-compose.yml config > /dev/null

# ------------------------------------------------------------------------------
# Test infrastructure lifecycle
# ------------------------------------------------------------------------------

phase1-sprint-01-services:
	$(TEST_COMPOSE) up -d $(TEST_SERVICES) --wait
	cd $(BACKEND_DIR) && $(WITH_TEST_ENV) $(ALEMBIC) upgrade head

phase1-sprint-01-down:
	$(TEST_COMPOSE) down

# ------------------------------------------------------------------------------
# Static analysis
# ------------------------------------------------------------------------------

phase1-sprint-01-static:
	cd $(BACKEND_DIR) && $(RUFF) check .
	cd $(BACKEND_DIR) && $(MYPY) .

phase1-sprint-01-format-check:
	cd $(BACKEND_DIR) && $(RUFF) format --check .

# ------------------------------------------------------------------------------
# Test suites
# ------------------------------------------------------------------------------

phase1-sprint-01-unit:
	cd $(BACKEND_DIR) && $(WITH_TEST_ENV) $(PYTEST) tests/unit -v

phase1-sprint-01-integration: phase1-sprint-01-services
	cd $(BACKEND_DIR) && $(WITH_TEST_ENV) $(PYTEST) tests/integration -v

phase1-sprint-01-full-test: phase1-sprint-01-services
	@mkdir -p $(TEST_REPORTS)
	cd $(BACKEND_DIR) && $(WITH_TEST_ENV) $(PYTEST) tests -v \
	    --junitxml=$(TEST_REPORTS)/sprint01.xml

# ------------------------------------------------------------------------------
# Restart persistence probe
# ------------------------------------------------------------------------------

phase1-sprint-01-restart: phase1-sprint-01-services
	@mkdir -p $(TEST_REPORTS)
	$(WITH_TEST_ENV) $(PYTHON) -m backend.scripts.restart_persistence_probe seed \
	    --state-file $(TEST_REPORTS)/restart-probe.json
	$(TEST_COMPOSE) restart $(TEST_SERVICES)
	$(TEST_COMPOSE) up -d $(TEST_SERVICES) --wait
	$(WITH_TEST_ENV) $(PYTHON) -m backend.scripts.restart_persistence_probe verify \
	    --state-file $(TEST_REPORTS)/restart-probe.json

# ------------------------------------------------------------------------------
# Container build smoke
# ------------------------------------------------------------------------------

phase1-sprint-01-image-build:
	docker build \
	    -f $(BACKEND_DIR)/Dockerfile \
	    -t mergenvision-backend:sprint01-check \
	    .
	docker run --rm mergenvision-backend:sprint01-check \
	    python -c "import app; print('backend-import-ok')"

# ------------------------------------------------------------------------------
# Full acceptance
# ------------------------------------------------------------------------------

phase1-sprint-01-acceptance: \
        phase1-sprint-01-preflight \
        phase1-sprint-01-services \
        phase1-sprint-01-static \
        phase1-sprint-01-format-check \
        phase1-sprint-01-unit \
        phase1-sprint-01-full-test \
        phase1-sprint-01-restart \
        phase1-sprint-01-image-build
	git diff --check

# ==============================================================================
# Sprint 002 — Offline Video Reference Correctness Laboratory
# ==============================================================================

LAB_DIR         := research/video_reference_lab
LAB_VENV        := $(PWD)/$(LAB_DIR)/.venv
LAB_PYTHON      := $(LAB_VENV)/bin/python
LAB_PIP         := $(LAB_PYTHON) -m pip
LAB_PYTEST      := $(LAB_PYTHON) -m pytest
LAB_RUFF        := $(LAB_PYTHON) -m ruff
LAB_MYPY        := $(LAB_PYTHON) -m mypy
LAB_MV          := $(LAB_VENV)/bin/mv-video-lab
LAB_REPORTS     := $(PWD)/test-reports/video-reference

# Default to CPU extra; override with VIDEO_LAB_EXTRA=cuda for CUDA runtime.
VIDEO_LAB_EXTRA ?= cpu

.PHONY: video-reference-install \
        video-reference-doctor \
        video-reference-unit \
        video-reference-smoke \
        video-reference-extract-friends \
        video-reference-replay-friends \
        video-reference-visualize-friends \
        video-reference-evaluate-friends \
        video-reference-benchmark \
        video-reference-acceptance \
        video-reference-friends

video-reference-install:
	@echo "==> Creating video reference lab environment..."
	python3.12 -m venv $(LAB_VENV)
	cd $(LAB_DIR) && $(LAB_PIP) install --upgrade pip wheel setuptools
	cd $(LAB_DIR) && $(LAB_PIP) install -e ".[dev,$(VIDEO_LAB_EXTRA)]"
	cd $(LAB_DIR) && $(LAB_PIP) check
	@echo "==> Video reference lab install complete."

video-reference-doctor:
	cd $(LAB_DIR) && $(LAB_MV) doctor --config configs/friends_baseline.yaml

video-reference-unit:
	@mkdir -p $(LAB_REPORTS)
	cd $(LAB_DIR) && $(LAB_PYTEST) tests/unit -v \
	    --junitxml=$(LAB_REPORTS)/unit.xml

video-reference-smoke:
	@mkdir -p $(LAB_REPORTS)
	cd $(LAB_DIR) && $(LAB_MV) run-friends \
	    --config configs/friends_baseline.yaml \
	    --max-frames 32 \
	    --run-dir $(LAB_REPORTS)/smoke

video-reference-extract-friends:
	cd $(LAB_DIR) && $(LAB_MV) run-friends \
	    --config configs/friends_baseline.yaml \
	    --stages extract

video-reference-replay-friends:
	cd $(LAB_DIR) && $(LAB_MV) run-friends \
	    --config configs/friends_baseline.yaml \
	    --stages replay

video-reference-visualize-friends:
	cd $(LAB_DIR) && $(LAB_MV) run-friends \
	    --config configs/friends_baseline.yaml \
	    --stages visualize

video-reference-evaluate-friends:
	cd $(LAB_DIR) && $(LAB_MV) evaluate \
	    --config configs/friends_baseline.yaml

video-reference-benchmark:
	cd $(LAB_DIR) && $(LAB_MV) benchmark \
	    --config configs/friends_baseline.yaml

video-reference-acceptance: video-reference-unit
	@mkdir -p $(LAB_REPORTS)
	cd $(LAB_DIR) && $(LAB_RUFF) check src tests
	cd $(LAB_DIR) && $(LAB_MYPY) src tests
	cd $(LAB_DIR) && $(LAB_RUFF) format --check src tests
	cd $(LAB_DIR) && $(LAB_PYTEST) tests -v \
	    --junitxml=$(LAB_REPORTS)/acceptance.xml
	git diff --check

video-reference-friends:
	cd $(LAB_DIR) && $(LAB_MV) run-friends \
	    --config configs/friends_baseline.yaml

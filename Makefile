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
LAB_VENV_CPU    := $(PWD)/$(LAB_DIR)/.venv-cpu
LAB_VENV_CUDA   := $(PWD)/$(LAB_DIR)/.venv-cuda
LAB_PYTHON      := $(LAB_VENV_CPU)/bin/python
LAB_PIP         := $(LAB_PYTHON) -m pip
LAB_PYTEST      := $(LAB_PYTHON) -m pytest
LAB_RUFF        := $(LAB_PYTHON) -m ruff
LAB_MYPY        := $(LAB_PYTHON) -m mypy
LAB_MV          := $(LAB_VENV_CPU)/bin/mv-video-lab
LAB_REPORTS     := $(PWD)/test-reports/video-reference

# Default config and environment.
VIDEO_LAB_CONFIG ?= configs/friends_baseline_cpu.yaml
VIDEO_LAB_EXTRA  ?= cpu

ifeq ($(VIDEO_LAB_EXTRA),cuda)
  LAB_PYTHON := $(LAB_VENV_CUDA)/bin/python
  LAB_PIP    := $(LAB_PYTHON) -m pip
  LAB_RUFF   := $(LAB_PYTHON) -m ruff
  LAB_MYPY   := $(LAB_PYTHON) -m mypy
  LAB_MV     := $(LAB_VENV_CUDA)/bin/mv-video-lab
endif

.PHONY: video-reference-clean-install-cpu \
        video-reference-clean-install-cuda \
        video-reference-cli-smoke \
        video-reference-unit \
        video-reference-synthetic-e2e \
        video-reference-artifact-integrity \
        video-reference-chunk-parity \
        video-reference-static \
        video-reference-ci \
        video-reference-models-cpu \
        video-reference-doctor-cpu \
        video-reference-real-model-smoke-cpu \
        video-reference-doctor-cuda \
        video-reference-real-model-smoke-cuda \
        video-reference-friends-smoke \
        video-reference-friends-extract \
        video-reference-friends-replay \
        video-reference-friends-visualize \
        video-reference-friends-evaluate \
        video-reference-friends-acceptance \
        video-reference-acceptance

# ------------------------------------------------------------------------------
# Installation
# ------------------------------------------------------------------------------

video-reference-clean-install-cpu:
	@echo "==> Creating video reference lab CPU environment..."
	rm -rf $(LAB_VENV_CPU)
	python3.12 -m venv $(LAB_VENV_CPU)
	cd $(LAB_DIR) && $(LAB_PIP) install --upgrade pip wheel setuptools
	cd $(LAB_DIR) && $(LAB_PIP) install -r requirements-cpu.lock
	cd $(LAB_DIR) && $(LAB_PIP) install -e ".[dev,cpu]" --no-deps
	cd $(LAB_DIR) && $(LAB_PIP) check
	cd $(LAB_DIR) && $(LAB_MV) --help > /dev/null
	@echo "==> CPU install complete."

video-reference-clean-install-cuda:
	@echo "==> Creating video reference lab CUDA environment..."
	rm -rf $(LAB_VENV_CUDA)
	python3.12 -m venv $(LAB_VENV_CUDA)
	cd $(LAB_DIR) && $(LAB_PIP) install --upgrade pip wheel setuptools
	cd $(LAB_DIR) && $(LAB_PIP) install -r requirements-cuda.lock
	cd $(LAB_DIR) && $(LAB_PIP) install -e ".[dev,cuda]" --no-deps
	cd $(LAB_DIR) && $(LAB_PIP) check
	cd $(LAB_DIR) && $(LAB_MV) --help > /dev/null
	@echo "==> CUDA install complete."

# ------------------------------------------------------------------------------
# Smoke / static / unit
# ------------------------------------------------------------------------------

video-reference-cli-smoke:
	cd $(LAB_DIR) && $(LAB_MV) --help
	cd $(LAB_DIR) && $(LAB_MV) doctor --config $(VIDEO_LAB_CONFIG) || true

video-reference-unit:
	@mkdir -p $(LAB_REPORTS)
	cd $(LAB_DIR) && $(LAB_PYTEST) tests/unit -v \
	    --junitxml=$(LAB_REPORTS)/unit.xml

video-reference-synthetic-e2e:
	@mkdir -p $(LAB_REPORTS)
	cd $(LAB_DIR) && $(LAB_PYTEST) tests/integration/test_synthetic_pipeline.py -v \
	    --junitxml=$(LAB_REPORTS)/synthetic-e2e.xml

video-reference-artifact-integrity:
	@mkdir -p $(LAB_REPORTS)
	cd $(LAB_DIR) && $(LAB_PYTEST) tests/integration/test_artifact_resume.py -v \
	    --junitxml=$(LAB_REPORTS)/artifact-integrity.xml

video-reference-chunk-parity:
	@mkdir -p $(LAB_REPORTS)
	cd $(LAB_DIR) && $(LAB_PYTEST) tests/unit/test_chunk_invariance.py -v \
	    --junitxml=$(LAB_REPORTS)/chunk-parity.xml

video-reference-static:
	cd $(LAB_DIR) && $(LAB_RUFF) check src tests
	cd $(LAB_DIR) && $(LAB_MYPY) src tests
	cd $(LAB_DIR) && $(LAB_RUFF) format --check src tests

video-reference-ci: video-reference-static video-reference-unit video-reference-synthetic-e2e video-reference-artifact-integrity video-reference-chunk-parity
	git diff --check

# ------------------------------------------------------------------------------
# Real model targets
# ------------------------------------------------------------------------------

video-reference-models-cpu:
	cd $(LAB_DIR) && $(LAB_MV) models acquire \
	    --name buffalo_l --provider cpu --allow-download

video-reference-doctor-cpu:
	cd $(LAB_DIR) && $(LAB_MV) doctor --config configs/friends_baseline_cpu.yaml

video-reference-real-model-smoke-cpu:
	cd $(LAB_DIR) && $(LAB_PYTEST) tests/integration/test_real_model_smoke.py -v

video-reference-doctor-cuda:
	cd $(LAB_DIR) && VIDEO_LAB_EXTRA=cuda $(LAB_MV) doctor --config configs/friends_baseline_cuda.yaml

video-reference-real-model-smoke-cuda:
	cd $(LAB_DIR) && VIDEO_LAB_EXTRA=cuda $(LAB_PYTEST) tests/integration/test_real_model_smoke.py -v

# ------------------------------------------------------------------------------
# Friends targets
# ------------------------------------------------------------------------------

video-reference-friends-smoke:
	@mkdir -p $(LAB_REPORTS)/friends-smoke
	cd $(LAB_DIR) && $(LAB_MV) run-friends \
	    --config $(VIDEO_LAB_CONFIG) \
	    --max-frames 32

video-reference-friends-extract:
	cd $(LAB_DIR) && $(LAB_MV) extract --config $(VIDEO_LAB_CONFIG)

video-reference-friends-replay:
	cd $(LAB_DIR) && $(LAB_MV) replay --config $(VIDEO_LAB_CONFIG)

video-reference-friends-visualize:
	cd $(LAB_DIR) && $(LAB_MV) visualize --config $(VIDEO_LAB_CONFIG)

video-reference-friends-evaluate:
	cd $(LAB_DIR) && $(LAB_MV) evaluate --config $(VIDEO_LAB_CONFIG)

video-reference-friends-acceptance:
	cd $(LAB_DIR) && $(LAB_MV) run-friends --config $(VIDEO_LAB_CONFIG)

video-reference-acceptance: video-reference-ci video-reference-doctor-cpu video-reference-real-model-smoke-cpu video-reference-friends-acceptance

# ==============================================================================
# Phase 2 — Complete Video Recognition Product
# ==============================================================================

PHASE2_PROJECT := mergenvision-p2-test
PHASE2_COMPOSE := docker compose -p $(PHASE2_PROJECT) -f $(COMPOSE_FILE)
FRONTEND_DIR   := frontend

.PHONY: phase2-services \
        phase2-step0-static phase2-step0-api-contract phase2-step0-storage \
        phase2-step0-failure phase2-step0-native phase2-step0-closure \
        phase2-migrations phase2-control-plane phase2-m3-worker-control \
        phase2-m4-device-pipeline phase2-m5-video-observation \
        phase2-m5-native-runtime-gate phase2-m5-build-engines phase2-m5-native-build \
        phase2-m5-gpu-decode-smoke phase2-m5-sequence-contract \
        phase2-m6-native-full-observation phase2-m6-track-template \
        phase2-m6-track-reconcile \
        phase2-m7-video-identity \
        phase2-m8-video-result \
        phase2-m8-ui-static phase2-m8-ui-unit phase2-m8-ui-build \
        phase2-m8-ui-playwright phase2-video-e2e-acceptance

phase2-services:
	$(PHASE2_COMPOSE) up -d $(TEST_SERVICES) --wait
	cd $(BACKEND_DIR) && $(WITH_TEST_ENV) $(ALEMBIC) upgrade head

phase2-step0-static:
	cd $(BACKEND_DIR) && $(RUFF) check app tests scripts
	cd $(BACKEND_DIR) && $(MYPY) app

phase2-step0-api-contract:
	cd $(BACKEND_DIR) && $(WITH_TEST_ENV) $(PYTEST) \
	    tests/unit/api/test_phase2_step0_contract.py \
	    tests/unit/api/test_health.py -q

phase2-step0-storage:
	cd $(BACKEND_DIR) && $(WITH_TEST_ENV) $(PYTEST) \
	    tests/integration/vectors/test_qdrant_model_version.py \
	    tests/integration/lifecycle/test_delete_detail_history.py -q

phase2-step0-failure:
	cd $(BACKEND_DIR) && $(WITH_TEST_ENV) $(PYTEST) \
	    tests/unit/services/test_image_orchestration.py \
	    tests/unit/api/test_image_validation.py -q

phase2-step0-native:
	cd $(BACKEND_DIR) && $(WITH_TEST_ENV) $(PYTEST) \
	    tests/native/test_image_runtime_surface.py \
	    tests/native/test_image_runtime_safety.py -q

phase2-step0-closure: phase2-services \
        phase2-step0-static phase2-step0-api-contract \
        phase2-step0-storage phase2-step0-failure phase2-step0-native
	git diff --check

phase2-migrations: phase2-services
	cd $(BACKEND_DIR) && $(WITH_TEST_ENV) $(ALEMBIC) upgrade head
	cd $(BACKEND_DIR) && $(WITH_TEST_ENV) $(PYTEST) \
	    tests/integration/persistence/test_migrations.py \
	    tests/integration/persistence/test_phase2_migrations.py -v

phase2-control-plane: phase2-services
	@echo "==> phase2-control-plane: upload/idempotency/failure/route safety"
	cd $(BACKEND_DIR) && $(WITH_TEST_ENV) $(PYTEST) \
	    tests/unit/persistence/test_tracked_model_modules.py \
	    tests/integration/video/test_upload_and_job.py -v
	@echo "==> phase2-control-plane passed"

phase2-m3-worker-control: phase2-services
	@echo "==> phase2-m3-worker-control: atomic claim/lease/heartbeat/retry/recovery"
	cd $(BACKEND_DIR) && $(WITH_TEST_ENV) $(PYTEST) \
	    tests/integration/video/test_job_queue.py -v
	@echo "==> phase2-m3-worker-control passed"

phase2-m4-device-pipeline: phase2-services
	@echo "==> phase2-m4-device-pipeline: common device-resident FacePipeline contract"
	cd $(BACKEND_DIR) && $(WITH_TEST_ENV) $(PYTEST) \
	    tests/unit/ports/test_face_pipeline_port.py \
	    tests/native/test_image_runtime_surface.py \
	    tests/native/test_image_runtime_safety.py -v
	@echo "==> phase2-m4-device-pipeline passed (host contract only; native GPU impl pending)"

phase2-m5-video-observation:
	@echo "==> phase2-m5-video-observation: contract only; real NVIDIA worker is blocked on native implementation"
	cd $(BACKEND_DIR) && $(WITH_TEST_ENV) $(PYTEST) \
	    tests/unit/contracts/test_video_observation_proto.py -v
	@echo "==> phase2-m5-video-observation contract present; real GPU smoke NOT_RUN"

phase2-m5-native-runtime-gate:
	@echo "==> phase2-m5-native-runtime-gate: DeepStream/TensorRT runtime + model/engine manifest"
	docker run --rm --gpus all \
	    -v $(PWD):/workspace \
	    -w /workspace \
	    -e MERGENVISION_REPO_ROOT=/workspace \
    --entrypoint python3 \
    mergenvision/deepstream-dev:9.0 \
    backend/scripts/inspect_video_runtime.py
	@echo "==> phase2-m5-native-runtime-gate passed"

phase2-m5-build-engines:
	@echo "==> building TensorRT engines inside DeepStream 9.0 container"
	docker run --rm --gpus all \
	    -v $(PWD):/workspace \
	    -w /workspace \
    --entrypoint python3 \
    mergenvision/deepstream-dev:9.0 \
    backend/scripts/build_engines.py \
--profile backend/config/model_profiles/retinaface_r50_glintr100_v1_deepstream9.json
	@echo "==> engines built"

phase2-m5-native-build: phase2-m5-native-runtime-gate
	@echo "==> building native video worker inside DeepStream 9.0 container"
	docker run --rm --gpus all \
	    -v $(PWD):/workspace \
	    -w /workspace \
	    --entrypoint bash \
	    mergenvision/deepstream-dev:9.0 \
	    -c "apt-get update >/dev/null 2>&1 && apt-get install -y --no-install-recommends libwebp-dev libzstd-dev >/dev/null 2>&1 && cmake -S backend/native/video_worker -B backend/native/video_worker/build && cmake --build backend/native/video_worker/build -j$(shell nproc)"
	@echo "==> native video worker built"

phase2-m5-gpu-decode-smoke: phase2-m5-native-build
	@echo "==> running real GPU decode smoke on Friends.mp4"
	docker run --rm --gpus all \
	    -v $(PWD):/workspace \
	    -w /workspace \
	    --entrypoint bash \
	    mergenvision/deepstream-dev:9.0 \
	    -c "unset USE_NEW_NVSTREAMMUX && backend/native/video_worker/build/decode_smoke --input test_videos/Friends.mp4 --all-frames"
	@echo "==> GPU decode smoke passed"

phase2-m5-sequence-contract: phase2-m5-native-runtime-gate
	@echo "==> building and running M5.1 sequence contract tests"
	docker run --rm --gpus all \
	    -v $(PWD):/workspace \
	    -w /workspace \
	    --entrypoint bash \
	    mergenvision/deepstream-dev:9.0 \
	    -c "cmake -S backend/native/video_worker -B backend/native/video_worker/build && cmake --build backend/native/video_worker/build --target sequence_contract -j$$(nproc) && backend/native/video_worker/build/sequence_contract"
	@echo "==> M5.1 sequence contract passed"

phase2-m6-native-full-observation: phase2-m5-native-build
	@echo "==> running full GPU observation pipeline on Friends.mp4 (RGBA alignment + GlintR100)"
	docker run --rm --gpus all \
	    -v $(PWD):/workspace \
	    -w /workspace \
	    --entrypoint bash \
	    mergenvision/deepstream-dev:9.0 \
	    -c "unset USE_NEW_NVSTREAMMUX && backend/native/video_worker/build/real_batching_smoke --input test_videos/Friends.mp4 --all-frames --detector-batch-size 16 --recognizer-batch-size 32"
	@echo "==> M6 native full observation passed"

phase2-m6-track-template: phase2-services
	@echo "==> phase2-m6-track-template: raw tracklet builder + quality template selector"
	cd $(BACKEND_DIR) && $(WITH_TEST_ENV) $(PYTEST) \
	    tests/unit/services/test_video_tracking_service.py -v
	@echo "==> phase2-m6-track-template passed"

phase2-m6-track-reconcile: phase2-services
	@echo "==> phase2-m6-track-reconcile: conservative raw-track reconciliation"
	cd $(BACKEND_DIR) && $(WITH_TEST_ENV) $(PYTEST) \
	    tests/unit/services/test_video_reconciliation_service.py -v
	@echo "==> phase2-m6-track-reconcile passed"

phase2-m7-video-identity: phase2-services
	@echo "==> phase2-m7-video-identity: canonical track -> persistent faceId resolution"
	cd $(BACKEND_DIR) && $(WITH_TEST_ENV) $(PYTEST) \
	    tests/unit/services/test_video_identity_resolution_service.py \
	    tests/unit/services/test_video_track_persistence_service.py \
	    tests/integration/video/test_video_identity_persistence.py -v
	@echo "==> phase2-m7-video-identity passed"

phase2-m8-video-result: phase2-services
	@echo "==> phase2-m8-video-result: job processing flow + people/appearances/timeline API"
	cd $(BACKEND_DIR) && $(WITH_TEST_ENV) $(PYTEST) \
	    tests/integration/video/test_video_processing_and_result_api.py -v
	@echo "==> phase2-m8-video-result passed"

phase2-m8-ui-static:
	@echo "==> phase2-m8-ui-static: TypeScript + lint"
	cd $(FRONTEND_DIR) && npm run typecheck
	cd $(FRONTEND_DIR) && npm run lint
	@echo "==> phase2-m8-ui-static passed"

phase2-m8-ui-unit:
	@echo "==> phase2-m8-ui-unit: frontend unit/component tests"
	cd $(FRONTEND_DIR) && npm test
	@echo "==> phase2-m8-ui-unit passed"

phase2-m8-ui-build:
	@echo "==> phase2-m8-ui-build: production build"
	cd $(FRONTEND_DIR) && npm run build
	@echo "==> phase2-m8-ui-build passed"

PHASE2_E2E_LOG := $(PWD)/test-reports/backend-e2e.log
PHASE2_E2E_PID := $(PWD)/test-reports/backend-e2e.pid

phase2-video-e2e-acceptance: phase2-services
	@echo "==> phase2-video-e2e-acceptance: real backend + frontend E2E"
	@mkdir -p $(TEST_REPORTS)
	@if curl -sf http://localhost:8090/health/live >/dev/null 2>&1; then \
	    echo "==> backend already running on port 8090"; \
	else \
	    echo "==> starting backend on port 8090"; \
	    cd $(BACKEND_DIR) && $(WITH_TEST_ENV) nohup $(PYTHON) -m uvicorn app.api.main:create_app --factory --host 0.0.0.0 --port 8090 > $(PHASE2_E2E_LOG) 2>&1 & echo $$! > $(PHASE2_E2E_PID); \
	fi
	@for i in $$(seq 1 60); do \
	    if curl -sf http://localhost:8090/health/live >/dev/null 2>&1; then echo "==> backend ready"; break; fi; \
	    if [ $$i -eq 60 ]; then echo "==> backend failed to start"; cat $(PHASE2_E2E_LOG); exit 1; fi; \
	    sleep 1; \
	done
	cd $(FRONTEND_DIR) && npx playwright test
	@echo "==> stopping managed backend (if any)"
	@if [ -f $(PHASE2_E2E_PID) ]; then kill `cat $(PHASE2_E2E_PID)` 2>/dev/null || true; rm -f $(PHASE2_E2E_PID); fi
	@echo "==> phase2-video-e2e-acceptance finished"

# ------------------------------------------------------------------------------
# MCP / Codebase Memory helpers
# ------------------------------------------------------------------------------
MCP_PROJECT     ?= home-user-Workspace-MergenVisionPhase2v2
MCP_REFRESH_SINCE ?= HEAD

mcp-context:
	@echo "==> mcp-context: $(QUERY)"
	python3 scripts/mcp_context.py "$(QUERY)" --project $(MCP_PROJECT)

mcp-refresh:
	@echo "==> mcp-refresh since $(MCP_REFRESH_SINCE)"
	python3 scripts/mcp_refresh.py --project $(MCP_PROJECT) --since $(MCP_REFRESH_SINCE)

mcp-impact:
	@echo "==> mcp-impact: $(TARGET)"
	python3 scripts/mcp_impact.py "$(TARGET)" --project $(MCP_PROJECT)

mcp-status:
	@python3 scripts/mcp_status.py --project $(MCP_PROJECT)

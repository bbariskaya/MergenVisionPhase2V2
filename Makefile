DOCKER_SERVICES := postgres minio qdrant
PYTHON := $(PWD)/backend/.venv/bin/python

.PHONY: phase1-sprint-01-static phase1-sprint-01-postgres phase1-sprint-01-minio phase1-sprint-01-qdrant phase1-sprint-01-lifecycle phase1-sprint-01-failure phase1-sprint-01-restart phase1-sprint-01-acceptance

phase1-sprint-01-static:
	cd backend && $(PYTHON) -m ruff check .
	cd backend && $(PYTHON) -m ruff format --check .
	cd backend && $(PYTHON) -m mypy .

phase1-sprint-01-postgres:
	docker compose up -d postgres --wait
	cd backend && $(PYTHON) -m alembic upgrade head
	cd backend && $(PYTHON) -m pytest tests/integration/persistence/test_migrations.py -v

phase1-sprint-01-minio:
	docker compose up -d minio --wait
	cd backend && $(PYTHON) -m pytest tests/integration/storage/test_minio_adapter.py -v

phase1-sprint-01-qdrant:
	docker compose up -d qdrant --wait
	cd backend && $(PYTHON) -m pytest tests/integration/vectors/test_qdrant_adapter.py -v

phase1-sprint-01-lifecycle:
	docker compose up -d $(DOCKER_SERVICES) --wait
	cd backend && $(PYTHON) -m alembic upgrade head
	cd backend && $(PYTHON) -m pytest tests/integration/lifecycle/test_identity_storage_lifecycle.py tests/integration/lifecycle/test_multiple_samples.py tests/integration/lifecycle/test_inactive_rejection.py -v

phase1-sprint-01-failure:
	docker compose up -d $(DOCKER_SERVICES) --wait
	cd backend && $(PYTHON) -m alembic upgrade head
	cd backend && $(PYTHON) -m pytest tests/integration/lifecycle/test_failure_paths.py -v

phase1-sprint-01-restart:
	docker compose up -d $(DOCKER_SERVICES) --wait
	cd backend && $(PYTHON) -m alembic upgrade head
	cd backend && $(PYTHON) -m pytest tests/integration/lifecycle/test_restart_persistence.py -v

phase1-sprint-01-acceptance: phase1-sprint-01-static
	cd backend && $(PYTHON) -m pytest tests/unit/test_domain_dependency_boundary.py -v
	cd backend && $(PYTHON) -m pytest tests/unit/domain -v
	$(MAKE) phase1-sprint-01-postgres
	$(MAKE) phase1-sprint-01-minio
	$(MAKE) phase1-sprint-01-qdrant
	$(MAKE) phase1-sprint-01-lifecycle
	$(MAKE) phase1-sprint-01-failure
	$(MAKE) phase1-sprint-01-restart
	cd backend && $(PYTHON) -m ruff check . && $(PYTHON) -m mypy .
	git diff --check

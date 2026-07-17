#!/usr/bin/env bash
set -e

cd /app
alembic upgrade head

exec uvicorn app.api.main:create_app --factory --host 0.0.0.0 --port 8000 "$@"

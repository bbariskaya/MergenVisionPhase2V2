"""Restart-persistence probe for Phase 1 Sprint 01.

Usage (from the repository root):

    set -a && source backend/.env.test && set +a
    python -m backend.scripts.restart_persistence_probe seed \
        --state-file test-reports/restart-probe.json

    docker compose -p mergenvision-s01-test -f docker-compose.test.yml restart \
        postgres-test minio-test qdrant-test
    docker compose -p mergenvision-s01-test -f docker-compose.test.yml up -d --wait

    python -m backend.scripts.restart_persistence_probe verify \
        --state-file test-reports/restart-probe.json

The probe writes only technical IDs and the MinIO object reference; it never
stores names, embeddings, or secrets in the state file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import asyncpg

from app.application.services.identity_storage_lifecycle_service import (
    IdentityStorageLifecycleService,
)
from app.domain.value_objects import BoundingBox
from app.infrastructure.persistence.sqlalchemy.session import async_session_maker
from app.infrastructure.persistence.sqlalchemy.unit_of_work import SqlAlchemyUnitOfWork
from app.infrastructure.storage.minio_adapter import MinIOObjectStore
from app.infrastructure.uuid7 import Uuid7Generator
from app.infrastructure.vectors.qdrant_adapter import QdrantVectorStore

DIMENSION = 512


def vector_a() -> list[float]:
    vec = [0.0] * DIMENSION
    vec[0] = 1.0
    return vec


BBOX = BoundingBox(x=0, y=0, width=16, height=16)
MATCH_THRESHOLD = 0.95


def _uow_factory() -> SqlAlchemyUnitOfWork:
    return SqlAlchemyUnitOfWork(async_session_maker)


async def _seed(state_file: Path) -> None:
    service = IdentityStorageLifecycleService(
        unit_of_work_factory=_uow_factory,
        object_store=MinIOObjectStore(),
        vector_store=QdrantVectorStore(),
        id_generator=Uuid7Generator(),
    )

    outcome = await service.resolve_or_create(
        crop_bytes=_crop_bytes(),
        embedding=vector_a(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )

    assert outcome.sample_id is not None

    async with SqlAlchemyUnitOfWork(async_session_maker) as uow:
        sample = await uow.face_samples.get_by_id(outcome.sample_id)
        assert sample is not None
        assert sample.object_key is not None
        bucket = sample.bucket
        object_key = sample.object_key

    state = {
        "process_id": str(outcome.process_id),
        "face_id": str(outcome.face_id),
        "sample_id": str(outcome.sample_id),
        "bucket": bucket,
        "object_key": object_key,
        "byte_size": len(_crop_bytes()),
    }

    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"seed: wrote {state_file}")


async def _verify(state_file: Path) -> None:
    state = json.loads(state_file.read_text(encoding="utf-8"))

    # PostgreSQL checks
    url = os.environ["DATABASE_URL"].replace("postgresql+asyncpg", "postgresql")
    conn = await asyncpg.connect(url)
    try:
        identity = await conn.fetchrow(
            "SELECT is_active, status FROM face_identity WHERE face_id = $1",
            state["face_id"],
        )
        assert identity is not None, "PostgreSQL identity missing"
        assert identity["is_active"] is True, "PostgreSQL identity not active"

        sample = await conn.fetchrow(
            "SELECT state, is_active, bucket, object_key FROM face_sample WHERE sample_id = $1",
            state["sample_id"],
        )
        assert sample is not None, "PostgreSQL sample missing"
        assert sample["state"] == "active", "PostgreSQL sample not active"
        assert sample["is_active"] is True, "PostgreSQL sample is_active false"
        assert sample["bucket"] == state["bucket"], "PostgreSQL bucket mismatch"
        assert sample["object_key"] == state["object_key"], "PostgreSQL object_key mismatch"

        process = await conn.fetchrow(
            "SELECT status FROM process_record WHERE process_id = $1",
            state["process_id"],
        )
        assert process is not None, "PostgreSQL process missing"
        assert process["status"] == "completed", "PostgreSQL process not completed"

        result_count = await conn.fetchval(
            "SELECT count(*) FROM recognition_result WHERE process_id = $1",
            state["process_id"],
        )
        assert result_count == 1, "PostgreSQL result count mismatch"
    finally:
        await conn.close()

    # MinIO check
    minio_store = MinIOObjectStore()
    stat = await minio_store.stat(state["object_key"])
    assert stat is not None, "MinIO object missing"
    assert stat.size == state["byte_size"], "MinIO size mismatch"
    assert stat.bucket == state["bucket"], "MinIO bucket mismatch"

    # Qdrant check
    qdrant_store = QdrantVectorStore()
    query_results = await qdrant_store.query(vector_a(), top_k=1)
    assert len(query_results) == 1, "Qdrant query returned no results"
    assert str(query_results[0].sample_id) == state["sample_id"], "Qdrant sample_id mismatch"
    assert str(query_results[0].face_id) == state["face_id"], "Qdrant face_id mismatch"

    print("verify: PostgreSQL, MinIO and Qdrant all consistent after restart")


def _crop_bytes() -> bytes:
    """Return the tiny valid WebP fixture used by the lifecycle tests."""
    path = Path(__file__).parents[1] / "tests" / "fixtures" / "valid_crop.webp"
    return path.read_bytes()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Restart persistence probe")
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed_parser = subparsers.add_parser("seed")
    seed_parser.add_argument("--state-file", required=True, type=Path)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--state-file", required=True, type=Path)

    args = parser.parse_args(argv)

    if args.command == "seed":
        asyncio.run(_seed(args.state_file))
    elif args.command == "verify":
        asyncio.run(_verify(args.state_file))


if __name__ == "__main__":
    main()

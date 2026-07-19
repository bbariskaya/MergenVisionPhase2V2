"""Read-only audit of identity storage consistency across PG/MinIO/Qdrant.

Usage (uses backend .env):
    backend/.venv/bin/python backend/scripts/audit_identity_storage.py

Reports are printed to stdout. No writes are performed.
"""

from __future__ import annotations

import asyncio
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import asyncpg
from minio import Minio
from qdrant_client import AsyncQdrantClient


def _load_env() -> None:
    env_path = Path(__file__).parents[1] / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key, value)


@dataclass
class AuditReport:
    pg_total: int = 0
    pg_active: int = 0
    pg_inactive: int = 0
    pg_pending: int = 0
    pg_failed: int = 0
    minio_objects: int = 0
    qdrant_points: int = 0
    missing_minio: list[str] = field(default_factory=list)
    missing_qdrant: list[str] = field(default_factory=list)
    orphan_minio_keys: list[str] = field(default_factory=list)
    orphan_qdrant_ids: list[str] = field(default_factory=list)
    qdrant_wrong_face: list[str] = field(default_factory=list)
    zero_byte_minio: list[str] = field(default_factory=list)
    minio_size_total: int = 0
    vector_dim_mismatch: list[str] = field(default_factory=list)

    def print_summary(self) -> None:
        print("=== Identity storage audit summary ===")
        print(f"PG total samples      : {self.pg_total}")
        print(f"  active              : {self.pg_active}")
        print(f"  inactive            : {self.pg_inactive}")
        print(f"  pending             : {self.pg_pending}")
        print(f"  failed              : {self.pg_failed}")
        print(f"MinIO objects under faces/ : {self.minio_objects}")
        print(f"Qdrant points              : {self.qdrant_points}")
        print(f"MinIO total bytes          : {self.minio_size_total}")
        print("---")
        print(f"PG samples missing MinIO object : {len(self.missing_minio)}")
        print(f"PG samples missing Qdrant point : {len(self.missing_qdrant)}")
        print(f"Orphan MinIO keys              : {len(self.orphan_minio_keys)}")
        print(f"Orphan Qdrant point IDs        : {len(self.orphan_qdrant_ids)}")
        print(f"Qdrant payload face_id mismatch : {len(self.qdrant_wrong_face)}")
        print(f"Zero-byte MinIO objects        : {len(self.zero_byte_minio)}")
        print(f"Vector dimension mismatch      : {len(self.vector_dim_mismatch)}")
        if self.missing_minio:
            print(f"First 5 missing MinIO sample IDs: {self.missing_minio[:5]}")
        if self.missing_qdrant:
            print(f"First 5 missing Qdrant sample IDs: {self.missing_qdrant[:5]}")
        if self.orphan_minio_keys:
            print(f"First 5 orphan MinIO keys: {self.orphan_minio_keys[:5]}")
        if self.orphan_qdrant_ids:
            print(f"First 5 orphan Qdrant IDs: {self.orphan_qdrant_ids[:5]}")
        if self.qdrant_wrong_face:
            print(f"First 5 face_id mismatches: {self.qdrant_wrong_face[:5]}")


async def _load_pg() -> tuple[dict[str, dict[str, object]], list[dict]]:
    database_url = os.environ["DATABASE_URL"].replace("postgresql+asyncpg", "postgresql")
    conn = await asyncpg.connect(database_url)
    try:
        rows = await conn.fetch(
            "SELECT sample_id, face_id, state, bucket, object_key FROM face_sample"
        )
    finally:
        await conn.close()

    by_id: dict[str, dict[str, object]] = {}
    for row in rows:
        by_id[str(row["sample_id"])] = {
            "sample_id": str(row["sample_id"]),
            "face_id": str(row["face_id"]),
            "state": row["state"],
            "bucket": row["bucket"],
            "object_key": row["object_key"],
        }
    return by_id, rows


def _load_minio_objects(bucket: str, endpoint: str, access_key: str, secret_key: str, secure: bool) -> dict[str, int]:
    client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
    objects: dict[str, int] = {}
    for obj in client.list_objects(bucket, prefix="faces/", recursive=True):
        objects[obj.object_name] = obj.size or 0
    return objects


async def _load_qdrant_points(url: str, collection_name: str) -> dict[str, tuple[str, int]]:
    client = AsyncQdrantClient(url=url)
    try:
        points: dict[str, tuple[str, int]] = {}
        offset: str | None = None
        while True:
            batch = await client.scroll(
                collection_name=collection_name,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for point in batch[0]:
                sample_id = point.payload.get("sample_id") if point.payload else None
                face_id = point.payload.get("face_id") if point.payload else None
                vector_dim = len(point.vector) if point.vector is not None else 0
                if sample_id:
                    points[str(sample_id)] = (str(face_id) if face_id else "", vector_dim)
            offset = batch[1]
            if offset is None:
                break
        return points
    finally:
        await client.close()


async def main() -> None:
    _load_env()
    database_url = os.environ["DATABASE_URL"]
    minio_endpoint = os.environ["MINIO_ENDPOINT"]
    minio_access_key = os.environ["MINIO_ACCESS_KEY"]
    minio_secret_key = os.environ["MINIO_SECRET_KEY"]
    minio_secure = os.environ.get("MINIO_SECURE", "false").lower() == "true"
    minio_bucket = os.environ["MINIO_BUCKET_NAME"]
    qdrant_url = os.environ["QDRANT_URL"]
    qdrant_collection = os.environ["QDRANT_COLLECTION_NAME"]

    report = AuditReport()

    pg_by_sample, pg_rows = await _load_pg()
    report.pg_total = len(pg_rows)
    for row in pg_rows:
        state = row["state"]
        if state == "active":
            report.pg_active += 1
        elif state == "inactive":
            report.pg_inactive += 1
        elif state == "pending":
            report.pg_pending += 1
        elif state == "failed":
            report.pg_failed += 1

    minio_objects, qdrant_points = await asyncio.gather(
        asyncio.to_thread(
            _load_minio_objects,
            minio_bucket,
            minio_endpoint,
            minio_access_key,
            minio_secret_key,
            minio_secure,
        ),
        _load_qdrant_points(qdrant_url, qdrant_collection),
    )

    report.minio_objects = len(minio_objects)
    report.qdrant_points = len(qdrant_points)
    report.minio_size_total = sum(minio_objects.values())

    expected_minio_keys: set[str] = set()
    expected_qdrant_ids: set[str] = set()
    for sample_id, sample in pg_by_sample.items():
        state = sample["state"]
        object_key = sample["object_key"]
        if state in ("active", "inactive") and object_key:
            expected_minio_keys.add(object_key)
        if state == "active":
            expected_qdrant_ids.add(sample_id)

    for sample_id in expected_qdrant_ids:
        sample = pg_by_sample[sample_id]
        if sample.get("object_key") and sample["object_key"] not in minio_objects:
            report.missing_minio.append(sample_id)
        size = minio_objects.get(sample["object_key"], 0)
        if size == 0 and sample.get("object_key"):
            report.zero_byte_minio.append(sample_id)

        if sample_id not in qdrant_points:
            report.missing_qdrant.append(sample_id)
        else:
            q_face_id, vector_dim = qdrant_points[sample_id]
            if q_face_id != sample["face_id"]:
                report.qdrant_wrong_face.append(f"{sample_id} qdrant_face={q_face_id} pg_face={sample['face_id']}")
            if vector_dim != 512:
                report.vector_dim_mismatch.append(f"{sample_id} dim={vector_dim}")

    for key in minio_objects:
        if key not in expected_minio_keys:
            report.orphan_minio_keys.append(key)

    for sample_id in qdrant_points:
        if sample_id not in expected_qdrant_ids:
            report.orphan_qdrant_ids.append(sample_id)

    report.print_summary()


if __name__ == "__main__":
    asyncio.run(main())

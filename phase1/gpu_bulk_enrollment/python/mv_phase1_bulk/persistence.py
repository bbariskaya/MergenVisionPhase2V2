"""Cross-store persistence orchestrator for Phase 1 bulk enrollment.

Lifecycle per sample:

1. PostgreSQL: person/face rows + ``face_sample`` in ``pending`` state.
2. MinIO: upload the original input JPEG, verifying size/SHA idempotency.
3. Qdrant: upsert the 512-D embedding with the required payload.
4. PostgreSQL: move ``face_sample`` to ``active``.

Failures are fail-closed: a sample that cannot be uploaded, vectorized, or
activated is marked ``failed`` in PostgreSQL.  Best-effort cleanup is attempted
for orphaned MinIO/Qdrant objects.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass

import numpy as np

from mv_phase1_bulk.identities import EnrolledSample, SubjectBundle
from mv_phase1_bulk.minio_store import MinioStore
from mv_phase1_bulk.postgres_store import PostgresStore
from mv_phase1_bulk.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PersistedSample:
    sample_id: str
    object_key: str
    embedding: np.ndarray


@dataclass(frozen=True)
class PersistedSubject:
    person_id: str
    face_id: str
    persisted: list[PersistedSample]
    failed: list[tuple[str, str]]


class PersistenceOrchestrator:
    """Coordinate PG -> MinIO -> Qdrant -> PG activation for one bundle."""

    def __init__(
        self,
        postgres: PostgresStore,
        minio: MinioStore,
        qdrant: QdrantStore,
        *,
        model_version: str,
    ) -> None:
        self._postgres = postgres
        self._minio = minio
        self._qdrant = qdrant
        self._model_version = model_version

    async def persist_bundle(
        self,
        bundle: SubjectBundle,
        rejected: list[tuple[str, str]] | None = None,
    ) -> PersistedSubject:
        """Persist one subject bundle across all three stores.

        ``rejected`` contains ``(sample_id, reason)`` tuples for samples that
        failed before persistence (e.g. no face or quarantine). They are marked
        ``failed`` in PostgreSQL so no row stays stuck in ``pending``.

        Returns a summary of which samples succeeded and which failed.
        """
        sample_records = [s.sample_record for s in bundle.samples]
        await self._postgres.prepare_enrollment([bundle.person], [bundle.face], sample_records)
        rejected = rejected or []

        if not bundle.samples:
            if rejected:
                await self._postgres.fail_samples_tx(rejected)
            return PersistedSubject(
                person_id=bundle.person.person_id,
                face_id=bundle.face.face_id,
                persisted=[],
                failed=rejected,
            )

        # 2. MinIO upload of the original input JPEG (idempotent, conflict-detecting).
        upload_items = [(bundle.face.face_id, s.sample_record.sample_id, s.image_bytes) for s in bundle.samples]
        upload_results = await self._minio.upload_many(upload_items, content_type="image/jpeg")

        successful: list[tuple[EnrolledSample, str]] = []
        failed: list[tuple[str, str]] = []
        for sample, result in zip(bundle.samples, upload_results, strict=True):
            if isinstance(result, BaseException):
                code = f"minio_upload:{type(result).__name__}"
                failed.append((sample.sample_record.sample_id, code))
                logger.warning("MinIO upload failed for %s: %s", sample.sample_record.sample_id, result)
                continue
            successful.append((sample, result.object_key))

        if not successful:
            await self._postgres.fail_samples_tx(failed)
            return PersistedSubject(
                person_id=bundle.person.person_id,
                face_id=bundle.face.face_id,
                persisted=[],
                failed=failed,
            )

        # 3. Qdrant batch upsert for successful uploads only.
        qdrant_items = [
            (sample.sample_record.sample_id, bundle.face.face_id, sample.embedding) for sample, _ in successful
        ]
        try:
            await self._qdrant.upsert_many(qdrant_items)
        except Exception as exc:
            logger.exception("Qdrant upsert failed for face %s", bundle.face.face_id)
            # Mark all successful uploads as failed and roll back MinIO objects.
            for sample, object_key in successful:
                failed.append((sample.sample_record.sample_id, f"qdrant_upsert:{type(exc).__name__}"))
                with contextlib.suppress(Exception):
                    await self._minio.delete_best_effort(object_key)
            await self._postgres.fail_samples_tx(failed)
            return PersistedSubject(
                person_id=bundle.person.person_id,
                face_id=bundle.face.face_id,
                persisted=[],
                failed=failed,
            )

        # 4. PostgreSQL activation.
        activations = [
            (sample.sample_record.sample_id, self._minio._bucket_name, object_key) for sample, object_key in successful
        ]
        try:
            await self._postgres.activate_samples_tx(activations)
        except Exception as exc:
            logger.exception("PostgreSQL activation failed for face %s", bundle.face.face_id)
            for sample, object_key in successful:
                failed.append((sample.sample_record.sample_id, f"pg_activate:{type(exc).__name__}"))
                with contextlib.suppress(Exception):
                    await self._minio.delete_best_effort(object_key)
                with contextlib.suppress(Exception):
                    await self._qdrant.delete_best_effort(sample.sample_record.sample_id)
            await self._postgres.fail_samples_tx(failed)
            return PersistedSubject(
                person_id=bundle.person.person_id,
                face_id=bundle.face.face_id,
                persisted=[],
                failed=failed,
            )

        persisted = [
            PersistedSample(
                sample_id=sample.sample_record.sample_id,
                object_key=object_key,
                embedding=sample.embedding,
            )
            for sample, object_key in successful
        ]

        all_failed = failed + rejected
        if all_failed:
            await self._postgres.fail_samples_tx(all_failed)

        return PersistedSubject(
            person_id=bundle.person.person_id,
            face_id=bundle.face.face_id,
            persisted=persisted,
            failed=all_failed,
        )

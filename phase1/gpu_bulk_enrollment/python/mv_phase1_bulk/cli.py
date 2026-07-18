"""Phase 1 bulk enrollment CLI."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer

from mv_phase1_bulk.admission import AdmissionError, AdmissionGate
from mv_phase1_bulk.config import Settings, get_settings, load_model_profile
from mv_phase1_bulk.identities import EnrolledSample, SubjectBundle, build_subject_bundles
from mv_phase1_bulk.ids import hmac_key_fingerprint
from mv_phase1_bulk.manifest import EnrollmentManifest
from mv_phase1_bulk.minio_store import MinioStore
from mv_phase1_bulk.persistence import PersistenceOrchestrator
from mv_phase1_bulk.postgres_store import PostgresStore
from mv_phase1_bulk.qdrant_store import QdrantStore
from mv_phase1_bulk.types import EnrollmentOutcome

app = typer.Typer(name="mv-phase1-bulk", help="Phase 1 isolated GPU bulk enrollment")


def _repo_root() -> Path:
    env_root = __import__("os").environ.get("MV_PHASE1_BULK_REPO_ROOT")
    if env_root:
        return Path(env_root)
    return Path(__file__).parents[4]


def _settings() -> Settings:
    return get_settings()


def _artifacts_dir() -> Path:
    return _repo_root() / ".artifacts" / "phase1_gpu_bulk_enrollment"


def _runs_dir() -> Path:
    runs = _artifacts_dir() / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    return runs


def _run_path(run_id: str) -> Path:
    return _runs_dir() / f"{run_id}.json"


@dataclass
class RunJournal:
    """Phase-1-local record of an enrollment run.

    No Phase 2 tables are touched; the journal lives under
    ``.artifacts/phase1_gpu_bulk_enrollment/runs/``.
    """

    run_id: str
    started_at: str
    completed_at: str
    source_namespace: str
    dataset_root: str
    manifest: str
    model_version: str
    preprocess_version: str
    gpu_device: int
    batch_size: int
    hmac_fingerprint: str
    outcomes: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_file(cls, path: Path) -> RunJournal:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _write_run_journal(journal: RunJournal) -> Path:
    path = _run_path(journal.run_id)
    path.write_text(journal.to_json(), encoding="utf-8")
    return path


def _load_run_journal(run_id: str) -> RunJournal:
    path = _run_path(run_id)
    if not path.exists():
        raise typer.BadParameter(f"run not found: {run_id}")
    return RunJournal.from_file(path)


@app.command()
def inspect_models(profile: Path = typer.Option(..., "--profile", help="Path to model_profile.json")) -> None:
    """Inspect ONNX models and report contract."""
    import onnx

    model_profile = load_model_profile(profile, repo_root=_repo_root())
    for key, model in model_profile["models"].items():
        onnx_path = Path(model["onnx_path"])
        if not onnx_path.is_absolute():
            onnx_path = _repo_root() / onnx_path
        data = onnx_path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        onnx_model = onnx.load(str(onnx_path))
        typer.echo(f"=== {key} ===")
        typer.echo(f"  path: {onnx_path}")
        typer.echo(f"  sha256: {sha}")
        typer.echo(f"  opset: {onnx_model.opset_import[0].version if onnx_model.opset_import else '?'}")
        for inp in onnx_model.graph.input:
            dims = [d.dim_param or (d.dim_value if d.dim_value else "?") for d in inp.type.tensor_type.shape.dim]
            typer.echo(f"  input: {inp.name} {onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type)} {dims}")
        for out in onnx_model.graph.output:
            dims = [d.dim_param or (d.dim_value if d.dim_value else "?") for d in out.type.tensor_type.shape.dim]
            typer.echo(f"  output: {out.name} {onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type)} {dims}")


@app.command()
def build_engines(
    profile: Path = typer.Option(..., "--profile", help="Path to model_profile.json"),
    workspace_mb: int = typer.Option(4096, "--workspace-mb"),
    device_id: int = typer.Option(0, "--device-id"),
) -> None:
    """Build Phase 1 TensorRT engines under the runtime-fingerprint directory."""
    from mv_phase1_bulk import engine_builder

    engine_builder.build_engines(
        profile,
        repo_root=_repo_root(),
        workspace_mb=workspace_mb,
        device_id=device_id,
    )
    typer.echo("build-engines: done")


@app.command()
def validate_manifest(
    dataset_root: Path = typer.Option(..., "--dataset-root"),
    manifest: Path = typer.Option(..., "--manifest"),
    require_sha256: bool = typer.Option(False, "--require-sha256"),
) -> None:
    """Validate enrollment manifest against schema and filesystem."""
    enrollment = EnrollmentManifest.from_file(
        dataset_root,
        manifest,
        require_sha256=require_sha256,
    )
    typer.echo(f"validate-manifest: subjects={len(enrollment)} images={enrollment.total_images()}")


@app.command()
def admit(skip_services: bool = typer.Option(False, "--skip-services")) -> None:
    """Check environment variables and service connectivity."""
    gate = AdmissionGate(_settings())
    report = asyncio.run(gate.admit(check_services=not skip_services))
    if report.ready:
        typer.echo("admit: ready")
        return
    for problem in report.problems:
        typer.echo(f"admit: {problem}", err=True)
    raise typer.Exit(code=1)


async def _run_enrollment(
    bundles: list[SubjectBundle],
    model_profile: dict[str, Any],
    source_namespace: str,
    gpu_device: int,
    batch_size: int,
) -> list[EnrollmentOutcome]:
    from mv_phase1_bulk.pipeline import GpuFacePipeline

    cfg = _settings()
    pg = PostgresStore(cfg.database_url)
    minio = MinioStore(
        endpoint=cfg.minio_endpoint,
        access_key=cfg.minio_access_key,
        secret_key=cfg.minio_secret_key,
        bucket_name=cfg.minio_bucket_name,
        secure=cfg.minio_secure,
    )
    qdrant = QdrantStore(
        url=cfg.qdrant_url,
        collection_name=cfg.qdrant_collection_name,
        model_version=cfg.model_version,
    )
    orchestrator = PersistenceOrchestrator(
        postgres=pg,
        minio=minio,
        qdrant=qdrant,
        model_version=cfg.model_version,
    )

    await pg.connect()
    pipeline: GpuFacePipeline | None = None
    try:
        pipeline = GpuFacePipeline(
            model_profile=model_profile,
            device_id=gpu_device,
        )
        pipeline.warmup()
        outcomes: list[EnrollmentOutcome] = []
        for bundle in bundles:
            image_bytes = [s.image_bytes for s in bundle.samples]
            source_keys = [s.sample_record.sample_id for s in bundle.samples]
            results = pipeline.extract_batch(
                image_bytes,
                source_keys=source_keys,
                max_batch=batch_size,
                multi_face_policy="quarantine",
            )

            accepted: list[EnrolledSample] = []
            rejected: list[tuple[str, str]] = []
            for sample, result in zip(bundle.samples, results, strict=True):
                if result.status == "accepted" and result.faces:
                    face = result.faces[0]
                    sample.set_extraction(face.crop_bytes, face.embedding)
                    accepted.append(sample)
                else:
                    reason = result.rejection_reason or result.status
                    sample.sample_record.failure_code = reason
                    rejected.append((sample.sample_record.sample_id, reason))

            bundle.samples = accepted
            persisted = await orchestrator.persist_bundle(bundle, rejected=rejected)

            persisted_ids = {s.sample_id for s in persisted.persisted}
            failed_ids = [sid for sid, _ in persisted.failed + rejected if sid not in persisted_ids]
            errors = [f"{sid}:{code}" for sid, code in persisted.failed + rejected]
            outcomes.append(
                EnrollmentOutcome(
                    external_subject_key=bundle.face.display_name,
                    face_id=bundle.face.face_id,
                    persisted_sample_ids=sorted(persisted_ids),
                    failed_sample_ids=sorted(failed_ids),
                    errors=sorted(errors),
                )
            )
            typer.echo(
                f"enrolled {bundle.face.display_name}: persisted={len(persisted.persisted)} failed={len(failed_ids)}"
            )
        return outcomes
    finally:
        if pipeline is not None:
            pipeline.close()
        await pg.close()


@app.command()
def enroll(
    dataset_root: Path = typer.Option(..., "--dataset-root"),
    manifest: Path = typer.Option(..., "--manifest"),
    source_namespace: str = typer.Option(..., "--source-namespace"),
    profile: Path = typer.Option(..., "--profile"),
    gpu_devices: str = typer.Option("0", "--gpu-devices"),
    batch_size: int = typer.Option(16, "--batch-size"),
    resume: bool = typer.Option(False, "--resume"),
    skip_admission: bool = typer.Option(False, "--skip-admission"),
) -> None:
    """Run bulk enrollment end-to-end and write a local run journal."""
    if resume:
        typer.echo("resume: not yet implemented; running full enrollment")

    if not skip_admission:
        gate = AdmissionGate(_settings())
        report = asyncio.run(gate.admit(check_services=True))
        if not report.ready:
            for problem in report.problems:
                typer.echo(f"admit: {problem}", err=True)
            raise AdmissionError("admission gate blocked enrollment")

    started_at = _now_iso()
    run_id = str(uuid.uuid4())
    repo_root = _repo_root()
    model_profile = load_model_profile(profile, repo_root=repo_root)
    enrollment_manifest = EnrollmentManifest.from_file(dataset_root, manifest)
    bundles = build_subject_bundles(
        enrollment_manifest,
        source_namespace=source_namespace,
        model_version=model_profile["model_version"],
        preprocess_version=model_profile["preprocess_version"],
    )

    gpu_device = int(gpu_devices.split(",")[0])
    outcomes = asyncio.run(
        _run_enrollment(
            bundles,
            model_profile,
            source_namespace,
            gpu_device,
            batch_size,
        )
    )
    total_persisted = sum(len(o.persisted_sample_ids) for o in outcomes)
    total_failed = sum(len(o.failed_sample_ids) for o in outcomes)
    journal = RunJournal(
        run_id=run_id,
        started_at=started_at,
        completed_at=_now_iso(),
        source_namespace=source_namespace,
        dataset_root=str(dataset_root.resolve()),
        manifest=str(manifest.resolve()),
        model_version=model_profile["model_version"],
        preprocess_version=model_profile["preprocess_version"],
        gpu_device=gpu_device,
        batch_size=batch_size,
        hmac_fingerprint=hmac_key_fingerprint(),
        outcomes=[o.to_compact_dict() for o in outcomes],
    )
    journal_path = _write_run_journal(journal)
    typer.echo(f"enroll: completed {len(outcomes)} subjects, {total_persisted} persisted, {total_failed} failed")
    typer.echo(f"run journal: {journal_path}")


async def _run_benchmark(
    dataset_root: Path,
    manifest_path: Path,
    profile: Path,
    gpu_device: int,
    batch_sizes: list[int],
    runs: int,
) -> None:
    from mv_phase1_bulk.pipeline import GpuFacePipeline

    model_profile = load_model_profile(profile, repo_root=_repo_root())
    enrollment_manifest = EnrollmentManifest.from_file(dataset_root, manifest_path)
    image_bytes: list[bytes] = []
    for record in enrollment_manifest:
        for image_path in record.image_paths:
            image_bytes.append(Path(image_path).read_bytes())

    typer.echo(f"benchmark: {len(image_bytes)} images from {len(enrollment_manifest)} subjects")
    pipeline = GpuFacePipeline(model_profile=model_profile, device_id=gpu_device)
    pipeline.warmup()
    try:
        # Warm-up on a small subset to amortize first-run overhead.
        warmup_count = min(len(image_bytes), 4)
        _ = pipeline.extract_batch(image_bytes[:warmup_count], max_batch=batch_sizes[0])

        for bs in batch_sizes:
            timings: list[float] = []
            for _ in range(runs):
                start = time.perf_counter()
                results = pipeline.extract_batch(image_bytes, max_batch=bs)
                elapsed = time.perf_counter() - start
                timings.append(elapsed)
            avg_s = sum(timings) / len(timings)
            throughput = len(image_bytes) / avg_s if avg_s > 0 else 0.0
            ms_per_img = 1000.0 / throughput if throughput > 0 else float("inf")
            statuses: dict[str, int] = {}
            results = pipeline.extract_batch(image_bytes, max_batch=bs)
            for r in results:
                statuses[r.status] = statuses.get(r.status, 0) + 1
            typer.echo(
                f"batch={bs:3d}  runs={runs}  avg={avg_s * 1000:.1f}ms  "
                f"throughput={throughput:.2f} img/s  ms/img={ms_per_img:.2f}  "
                f"statuses={statuses}"
            )
    finally:
        pipeline.close()


@app.command()
def benchmark(
    dataset_root: Path = typer.Option(..., "--dataset-root"),
    manifest: Path = typer.Option(..., "--manifest"),
    profile: Path = typer.Option(..., "--profile"),
    gpu_devices: str = typer.Option("0", "--gpu-devices"),
    batch_matrix: str = typer.Option("1,2,4,8,16", "--batch-matrix"),
    runs: int = typer.Option(3, "--runs"),
) -> None:
    """Run inference-only benchmark matrix across batch sizes."""
    try:
        batch_sizes = [int(x.strip()) for x in batch_matrix.split(",") if x.strip()]
    except ValueError as exc:
        raise typer.BadParameter(f"--batch-matrix must be comma-separated integers: {exc}") from exc
    if not batch_sizes:
        raise typer.BadParameter("--batch-matrix must contain at least one batch size")
    gpu_device = int(gpu_devices.split(",")[0])
    asyncio.run(
        _run_benchmark(
            dataset_root=dataset_root,
            manifest_path=manifest,
            profile=profile,
            gpu_device=gpu_device,
            batch_sizes=batch_sizes,
            runs=runs,
        )
    )


async def _run_reconcile(run_id: str) -> dict[str, Any]:
    journal = _load_run_journal(run_id)
    cfg = _settings()
    pg = PostgresStore(cfg.database_url)
    minio = MinioStore(
        endpoint=cfg.minio_endpoint,
        access_key=cfg.minio_access_key,
        secret_key=cfg.minio_secret_key,
        bucket_name=cfg.minio_bucket_name,
        secure=cfg.minio_secure,
    )
    qdrant = QdrantStore(
        url=cfg.qdrant_url,
        collection_name=cfg.qdrant_collection_name,
        model_version=cfg.model_version,
    )

    await pg.connect()
    try:
        face_ids = [o["face_id"] for o in journal.outcomes]
        db_samples = await pg.get_samples_for_face_ids(face_ids)
        db_by_sample = {s.sample_id: s for s in db_samples}

        checks: list[dict[str, Any]] = []
        for outcome in journal.outcomes:
            face_id = outcome["face_id"]
            for sample_id in outcome["persisted_sample_ids"]:
                record = db_by_sample.get(sample_id)
                pg_ok = record is not None and record.state == "active" and record.is_active
                object_key = record.object_key if record else None
                minio_ok = False
                if object_key:
                    stat = await minio.stat(object_key)
                    minio_ok = stat is not None and stat.size is not None and stat.size > 0
                qdrant_ok = False
                try:
                    point = await qdrant.retrieve(sample_id)
                    qdrant_ok = point is not None
                except Exception:
                    qdrant_ok = False
                checks.append(
                    {
                        "sample_id": sample_id,
                        "face_id": face_id,
                        "object_key": object_key,
                        "pg": pg_ok,
                        "minio": minio_ok,
                        "qdrant": qdrant_ok,
                    }
                )

        failed = [c for c in checks if not (c["pg"] and c["minio"] and c["qdrant"])]
        return {"run_id": run_id, "total": len(checks), "ok": len(checks) - len(failed), "failed": failed}
    finally:
        await pg.close()


@app.command()
def reconcile(run_id: str = typer.Option(..., "--run-id")) -> None:
    """Reconcile a previous run across PostgreSQL, MinIO, and Qdrant."""
    result = asyncio.run(_run_reconcile(run_id))
    typer.echo(f"reconcile: run_id={result['run_id']} total={result['total']} ok={result['ok']}")
    for failure in result["failed"]:
        typer.echo(
            f"  MISMATCH sample={failure['sample_id']} face={failure['face_id']} "
            f"pg={failure['pg']} minio={failure['minio']} qdrant={failure['qdrant']}"
        )
    if result["failed"]:
        raise typer.Exit(code=1)


@app.command()
def report(run_id: str = typer.Option(..., "--run-id")) -> None:
    """Generate a run report from the local run journal."""
    journal = _load_run_journal(run_id)
    total_subjects = len(journal.outcomes)
    total_persisted = sum(len(o["persisted_sample_ids"]) for o in journal.outcomes)
    total_failed = sum(len(o.get("failed_sample_ids", [])) for o in journal.outcomes)
    started = datetime.fromisoformat(journal.started_at)
    completed = datetime.fromisoformat(journal.completed_at)
    duration_s = (completed - started).total_seconds()

    status_counts: dict[str, int] = {}
    for outcome in journal.outcomes:
        for error in outcome.get("errors", []):
            # error format is "sample_id:status_or_code"
            code = error.split(":", 1)[-1] if ":" in error else error
            status_counts[code] = status_counts.get(code, 0) + 1

    typer.echo(f"=== Run {run_id} ===")
    typer.echo(f"source_namespace: {journal.source_namespace}")
    typer.echo(f"dataset: {journal.dataset_root}")
    typer.echo(f"manifest: {journal.manifest}")
    typer.echo(f"model_version: {journal.model_version}")
    typer.echo(f"preprocess_version: {journal.preprocess_version}")
    typer.echo(f"gpu_device: {journal.gpu_device}")
    typer.echo(f"batch_size: {journal.batch_size}")
    typer.echo(f"hmac_fingerprint: {journal.hmac_fingerprint}")
    typer.echo(f"started: {journal.started_at}")
    typer.echo(f"completed: {journal.completed_at}")
    typer.echo(f"duration: {duration_s:.2f}s")
    typer.echo(f"subjects: {total_subjects}")
    typer.echo(f"samples persisted: {total_persisted}")
    typer.echo(f"samples failed: {total_failed}")
    if status_counts:
        typer.echo("failure breakdown:")
        for code, count in sorted(status_counts.items()):
            typer.echo(f"  {code}: {count}")


if __name__ == "__main__":
    app()

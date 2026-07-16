"""Typer CLI for the offline video reference lab.

This module intentionally imports only the infrastructure needed for each
command. Heavy ML modules are loaded lazily inside command bodies so that
`--help` and `doctor` work even when no model is present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NoReturn

import typer
from rich.console import Console

from mergenvision_video_lab.config import LabConfig, load_config, resolve_repo_relative_path
from mergenvision_video_lab.errors import ErrorCode, LabError, ModelArtifactError
from mergenvision_video_lab.model_inventory import available_providers

app = typer.Typer(
    name="mv-video-lab",
    help="MergenVision offline video reference correctness laboratory.",
    no_args_is_help=True,
)
console = Console()


def _exit(code: ErrorCode, message: str, details: dict[str, Any] | None = None) -> NoReturn:
    console.print(f"[red]{code.value}[/red]: {message}")
    if details:
        console.print_json(json.dumps(details, default=str))
    raise typer.Exit(code=1)


# ------------------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------------------


def _load_config(config: str) -> LabConfig:
    try:
        return load_config(config)
    except LabError as exc:
        _exit(exc.code, exc.message, exc.details)


def _resolve_video_path(cfg: LabConfig) -> Path:
    return resolve_repo_relative_path(cfg.video.path)


# ------------------------------------------------------------------------------
# doctor
# ------------------------------------------------------------------------------


@app.command()
def doctor(
    config: str = typer.Option("configs/friends_baseline_cpu.yaml", "--config"),
) -> None:
    """Check environment, dependencies, input video, and model presence."""
    cfg = _load_config(config)

    console.print(f"[green]config loaded[/green]: {config}")
    video_path = _resolve_video_path(cfg)
    console.print(
        f"video path: {cfg.video.path} (resolved={video_path}, exists={video_path.exists()})"
    )
    if not video_path.exists():
        _exit(ErrorCode.BLOCKED_INPUT_VIDEO, f"input video not found: {video_path}")

    # Package versions.
    from importlib.metadata import version as pkg_version

    packages = [
        "numpy",
        "scipy",
        "av",
        "opencv-python-headless",
        "onnx",
        "onnxruntime",
        "insightface",
        "pydantic",
        "typer",
    ]
    versions = {}
    for name in packages:
        try:
            versions[name] = pkg_version(name)
        except Exception:
            versions[name] = "not installed"
    console.print_json(json.dumps(versions, indent=2, sort_keys=True))

    # Provider availability.
    providers = available_providers()
    console.print(f"ONNX Runtime providers available: {providers}")

    requested = cfg.oracle.provider
    if requested == "cuda" and "CUDAExecutionProvider" not in providers:
        _exit(
            ErrorCode.BLOCKED_REFERENCE_ENVIRONMENT,
            "CUDA provider requested but not available",
            {"available": providers},
        )

    # Optional model presence check (fail-closed if a model pack is configured).
    if cfg.oracle.model_pack:
        from mergenvision_video_lab.oracle.insightface_oracle import InsightFaceOracle

        try:
            oracle = InsightFaceOracle(
                model_root=cfg.oracle.model_root,
                model_pack=cfg.oracle.model_pack,
                requested_provider=cfg.oracle.provider,
                allow_cpu_fallback=cfg.oracle.allow_cpu_fallback,
                det_size=tuple(cfg.oracle.det_size),
            )
            console.print(f"model pack: {cfg.oracle.model_pack}")
            console.print(f"detector: {oracle.detector_contract.basename}")
            console.print(f"recognizer: {oracle.recognizer_contract.basename}")
            console.print(f"actual providers: {oracle._actual_providers}")
        except ModelArtifactError as exc:
            _exit(exc.code, exc.message, exc.details)

    console.print("[green]doctor OK[/green]")


# ------------------------------------------------------------------------------
# models
# ------------------------------------------------------------------------------


models_app = typer.Typer(name="models", help="Model acquisition and inspection.")
app.add_typer(models_app)


@models_app.command("inspect")
def models_inspect(
    config: str = typer.Option("configs/friends_baseline_cpu.yaml", "--config"),
) -> None:
    """Print local model inventory and ONNX contracts."""
    cfg = _load_config(config)
    from mergenvision_video_lab.oracle.insightface_oracle import InsightFaceOracle

    try:
        oracle = InsightFaceOracle(
            model_root=cfg.oracle.model_root,
            model_pack=cfg.oracle.model_pack,
            requested_provider=cfg.oracle.provider,
            allow_cpu_fallback=cfg.oracle.allow_cpu_fallback,
            det_size=tuple(cfg.oracle.det_size),
        )
    except LabError as exc:
        _exit(exc.code, exc.message, exc.details)

    console.print_json(json.dumps(oracle.detector_contract.model_dump(mode="json")))
    console.print_json(json.dumps(oracle.recognizer_contract.model_dump(mode="json")))


@models_app.command("acquire")
def models_acquire(
    name: str = typer.Option(..., "--name"),
    provider: str = typer.Option("cpu", "--provider"),
    allow_download: bool = typer.Option(False, "--allow-download"),
) -> None:
    """Acquire an official InsightFace model pack into a Git-ignored cache."""
    if not allow_download:
        _exit(
            ErrorCode.BLOCKED_MODEL_ARTIFACT,
            "Downloads are disabled by default. Use --allow-download to authorize once.",
        )
    allowed = {"buffalo_l"}
    if name not in allowed:
        _exit(
            ErrorCode.CONFIG_INVALID,
            f"Only official packs {allowed} are authorized.",
        )

    from mergenvision_video_lab.model_manager import acquire_model_pack

    try:
        manifest = acquire_model_pack(name=name, provider=provider)
    except LabError as exc:
        _exit(exc.code, exc.message, exc.details)

    console.print(f"[green]acquired[/green]: {name}")
    console.print_json(json.dumps(manifest, indent=2, default=str))


# ------------------------------------------------------------------------------
# Pipeline commands
# ------------------------------------------------------------------------------


def _run_stage(name: str, fn: Any) -> None:
    try:
        result = fn()
    except LabError as exc:
        _exit(exc.code, exc.message, exc.details)
    console.print(f"[green]{name} OK[/green]")
    if result:
        console.print_json(json.dumps(result, indent=2, default=str))


@app.command()
def extract(
    config: str = typer.Option("configs/friends_baseline_cpu.yaml", "--config"),
    max_frames: int | None = typer.Option(None, "--max-frames"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Run full reference extraction or resume if valid artifacts exist."""
    cfg = _load_config(config)
    if max_frames is not None:
        cfg.video.max_frames = max_frames
    from mergenvision_video_lab.pipeline import stage_extract

    _run_stage("extract", lambda: stage_extract(cfg, force=force))


@app.command()
def validate(
    config: str = typer.Option("configs/friends_baseline_cpu.yaml", "--config"),
) -> None:
    """Validate frozen artifacts."""
    cfg = _load_config(config)
    from mergenvision_video_lab.pipeline import stage_validate

    _run_stage("validate", lambda: stage_validate(cfg))


@app.command()
def replay(
    config: str = typer.Option("configs/friends_baseline_cpu.yaml", "--config"),
    strategy: str = typer.Option("byte_iou", "--strategy"),
    chunk_size: int = typer.Option(1, "--chunk-size"),
) -> None:
    """Replay frozen observations through a tracker."""
    cfg = _load_config(config)
    from mergenvision_video_lab.pipeline import stage_replay

    _run_stage("replay", lambda: stage_replay(cfg, strategy=strategy, chunk_size=chunk_size))


@app.command()
def templates(
    config: str = typer.Option("configs/friends_baseline_cpu.yaml", "--config"),
    strategy: str = typer.Option("byte_iou", "--strategy"),
) -> None:
    """Build tracklet templates."""
    cfg = _load_config(config)
    from mergenvision_video_lab.pipeline import stage_templates

    _run_stage("templates", lambda: stage_templates(cfg, strategy=strategy))


@app.command()
def reconcile(
    config: str = typer.Option("configs/friends_baseline_cpu.yaml", "--config"),
    strategy: str = typer.Option("byte_iou", "--strategy"),
) -> None:
    """Reconcile raw tracklets into canonical tracks."""
    cfg = _load_config(config)
    from mergenvision_video_lab.pipeline import stage_reconcile

    _run_stage("reconcile", lambda: stage_reconcile(cfg, strategy=strategy))


@app.command()
def gallery(
    config: str = typer.Option("configs/friends_baseline_cpu.yaml", "--config"),
    strategy: str = typer.Option("byte_iou", "--strategy"),
) -> None:
    """Build gallery identity templates and annotate canonical tracks."""
    cfg = _load_config(config)
    from mergenvision_video_lab.pipeline import stage_gallery

    _run_stage("gallery", lambda: stage_gallery(cfg, strategy=strategy))


@app.command()
def evaluate(
    config: str = typer.Option("configs/friends_baseline_cpu.yaml", "--config"),
    strategy: str = typer.Option("byte_iou", "--strategy"),
) -> None:
    """Evaluate tracking and identity metrics."""
    cfg = _load_config(config)
    from mergenvision_video_lab.pipeline import stage_evaluate

    _run_stage("evaluate", lambda: stage_evaluate(cfg, strategy=strategy))


@app.command()
def visualize(
    config: str = typer.Option("configs/friends_baseline_cpu.yaml", "--config"),
    strategy: str = typer.Option("byte_iou", "--strategy"),
) -> None:
    """Generate contact sheets, histograms, timeline, and overlay."""
    cfg = _load_config(config)
    from mergenvision_video_lab.pipeline import stage_visualize

    _run_stage("visualize", lambda: stage_visualize(cfg, strategy=strategy))


@app.command()
def benchmark(
    config: str = typer.Option("configs/friends_baseline_cpu.yaml", "--config"),
    strategy: str = typer.Option("byte_iou", "--strategy"),
) -> None:
    """Benchmark extraction and frozen replay."""
    cfg = _load_config(config)
    from mergenvision_video_lab.pipeline import stage_benchmark

    _run_stage("benchmark", lambda: stage_benchmark(cfg, strategy=strategy))


@app.command("run-friends")
def run_friends(
    config: str = typer.Option("configs/friends_baseline_cpu.yaml", "--config"),
    max_frames: int | None = typer.Option(None, "--max-frames"),
    stages: str | None = typer.Option(None, "--stages"),
) -> None:
    """Run the full Friends reference lab end-to-end."""
    cfg = _load_config(config)
    selected = stages.split(",") if stages else None
    from mergenvision_video_lab.pipeline import run_friends as pipeline_run_friends

    try:
        results = pipeline_run_friends(cfg, max_frames=max_frames, stages=selected)
    except LabError as exc:
        _exit(exc.code, exc.message, exc.details)

    console.print("[green]run-friends complete[/green]")
    console.print_json(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    app()

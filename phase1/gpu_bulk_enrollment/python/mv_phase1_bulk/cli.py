"""Phase 1 bulk enrollment CLI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import onnx
import typer

from mv_phase1_bulk import engine_builder

app = typer.Typer(name="mv-phase1-bulk", help="Phase 1 isolated GPU bulk enrollment")


def _repo_root() -> Path:
    return Path(__file__).parents[4]


@app.command()
def inspect_models(profile: Path = typer.Option(..., "--profile", help="Path to model_profile.json")) -> None:
    """Inspect ONNX models and report contract."""
    with profile.open("r", encoding="utf-8") as f:
        model_profile = json.load(f)

    repo_root = _repo_root()
    for key, model in model_profile["models"].items():
        onnx_path = repo_root / model["onnx_path"]
        data = onnx_path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        onnx_model = onnx.load(str(onnx_path))
        typer.echo(f"=== {key} ===")
        typer.echo(f"  path: {onnx_path}")
        typer.echo(f"  sha256: {sha}")
        typer.echo(f"  opset: {onnx_model.opset_import[0].version if onnx_model.opset_import else '?'}")
        for inp in onnx_model.graph.input:
            dims = [d.dim_param or (d.dim_value if d.dim_value else '?') for d in inp.type.tensor_type.shape.dim]
            typer.echo(f"  input: {inp.name} {onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type)} {dims}")
        for out in onnx_model.graph.output:
            dims = [d.dim_param or (d.dim_value if d.dim_value else '?') for d in out.type.tensor_type.shape.dim]
            typer.echo(f"  output: {out.name} {onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type)} {dims}")


@app.command()
def build_engines(
    profile: Path = typer.Option(..., "--profile", help="Path to model_profile.json"),
    workspace_mb: int = typer.Option(4096, "--workspace-mb"),
) -> None:
    """Build Phase 1 TensorRT engines."""
    engine_builder.build_engines(profile, workspace_mb=workspace_mb)
    typer.echo("build-engines: done")


@app.command()
def validate_manifest(
    dataset_root: Path = typer.Option(..., "--dataset-root"),
    manifest: Path = typer.Option(..., "--manifest"),
) -> None:
    """Validate enrollment manifest against schema and filesystem."""
    typer.echo(f"validate-manifest: dataset_root={dataset_root} manifest={manifest}")


@app.command()
def enroll(
    dataset_root: Path = typer.Option(..., "--dataset-root"),
    manifest: Path = typer.Option(..., "--manifest"),
    workers: int = typer.Option(1, "--workers"),
    gpu_devices: str = typer.Option("0", "--gpu-devices"),
    batch_size: int = typer.Option(16, "--batch-size"),
    resume: bool = typer.Option(False, "--resume"),
) -> None:
    """Run bulk enrollment."""
    typer.echo(
        f"enroll: dataset_root={dataset_root} manifest={manifest} "
        f"workers={workers} gpu_devices={gpu_devices} batch_size={batch_size} resume={resume}"
    )


@app.command()
def benchmark(
    dataset_root: Path = typer.Option(..., "--dataset-root"),
    manifest: Path = typer.Option(..., "--manifest"),
    gpu_devices: str = typer.Option("0", "--gpu-devices"),
    batch_matrix: str = typer.Option("1,2,4,8,16", "--batch-matrix"),
    runs: int = typer.Option(3, "--runs"),
) -> None:
    """Run benchmark matrix."""
    typer.echo(
        f"benchmark: dataset_root={dataset_root} manifest={manifest} "
        f"gpu_devices={gpu_devices} batch_matrix={batch_matrix} runs={runs}"
    )


@app.command()
def reconcile(run_id: str = typer.Option(..., "--run-id")) -> None:
    """Reconcile a previous run."""
    typer.echo(f"reconcile: run_id={run_id}")


@app.command()
def report(run_id: str = typer.Option(..., "--run-id")) -> None:
    """Generate run report."""
    typer.echo(f"report: run_id={run_id}")


if __name__ == "__main__":
    app()

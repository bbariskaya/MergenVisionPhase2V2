"""FastAPI backend for the sparse anchor annotator UI.

Serves video with Range support, overlay metadata, frozen tracklets, and
reads/writes the ground-truth YAML checkpoint. This module must stay free of
business logic beyond adapter helpers; the annotation authority is the YAML
file written by the frontend.
"""

from __future__ import annotations

import json
import mimetypes
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from mergenvision_video_lab.artifact_store import ArtifactStore
from mergenvision_video_lab.config import LabConfig, resolve_repo_relative_path, resolve_run_dir
from mergenvision_video_lab.contracts import GroundTruth, GroundTruthAnchor
from mergenvision_video_lab.ground_truth import (
    build_ground_truth_template,
    load_ground_truth,
    resolve_anchor_observations,
)


class AnchorSave(BaseModel):
    """Body for saving an anchor list."""

    anchors: list[GroundTruthAnchor]


class _RunContext(BaseModel):
    """Paths and helpers for one resolved reference run."""

    run_dir: Path
    strategy: str
    store: ArtifactStore
    video_path: Path
    overlay_path: Path
    tracklets_path: Path
    canonical_path: Path
    gt_path: Path

    model_config = ConfigDict(arbitrary_types_allowed=True)


def _make_context(config: LabConfig, strategy: str) -> _RunContext:
    run_dir = resolve_run_dir(config)
    if not run_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"run directory not found: {run_dir}. Run extraction first.",
        )
    store = ArtifactStore(run_dir)
    store.read_manifest()  # fail early if manifest missing
    video_path = resolve_repo_relative_path(config.video.path)
    return _RunContext(
        run_dir=run_dir,
        strategy=strategy,
        store=store,
        video_path=video_path,
        overlay_path=run_dir / "visual" / "overlay.jsonl",
        tracklets_path=run_dir / "replay" / strategy / "tracklets.jsonl",
        canonical_path=run_dir / "tracks" / "canonical_tracks.jsonl",
        gt_path=run_dir / "ground_truth.yaml",
    )


def _load_overlay(overlay_path: Path) -> dict[int, list[dict[str, Any]]]:
    by_frame: dict[int, list[dict[str, Any]]] = {}
    with open(overlay_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            by_frame[record["frame_index"]] = record["faces"]
    return by_frame


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_canonical_tracks(canonical_path: Path) -> list[dict[str, Any]]:
    if not canonical_path.exists():
        return []
    with open(canonical_path, encoding="utf-8") as f:
        first = f.read(1)
    if first == "{":
        with open(canonical_path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    return _load_jsonl(canonical_path)


def _load_gt(context: _RunContext) -> GroundTruth:
    if context.gt_path.exists():
        return load_ground_truth(context.gt_path)
    manifest = context.store.read_manifest()
    return build_ground_truth_template(manifest.video_sha256)


def _save_gt(gt_path: Path, gt: GroundTruth) -> None:
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = gt.model_dump(mode="json")
    with open(gt_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def _range_response(video_path: Path, request: Request) -> StreamingResponse:
    """Return the video file, supporting HTTP Range requests for browser seeking."""
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"video not found: {video_path}")
    file_size = video_path.stat().st_size
    media_type, _ = mimetypes.guess_type(str(video_path))
    if media_type is None:
        media_type = "video/mp4"

    range_header = request.headers.get("range")
    if range_header is None:
        return StreamingResponse(
            video_path.open("rb"),
            media_type=media_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
            },
        )

    try:
        unit, _, spec = range_header.partition("=")
        if unit.strip().lower() != "bytes":
            raise ValueError("only bytes ranges supported")
        start_str, _, end_str = spec.partition("-")
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1
        if start < 0 or end >= file_size or start > end:
            raise ValueError("invalid byte range")
    except Exception as exc:
        raise HTTPException(
            status_code=416, detail=f"invalid range: {range_header} ({exc})"
        ) from exc

    content_length = end - start + 1

    def _stream() -> Iterator[bytes]:
        with video_path.open("rb") as f:
            f.seek(start)
            remaining = content_length
            chunk_size = 64 * 1024
            while remaining > 0:
                read_size = min(chunk_size, remaining)
                data = f.read(read_size)
                if not data:
                    break
                yield data
                remaining -= len(data)

    return StreamingResponse(
        _stream(),
        media_type=media_type,
        status_code=206,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
            "Content-Range": f"bytes {start}-{end}/{file_size}",
        },
    )


def create_app(config: LabConfig, strategy: str) -> FastAPI:
    """Build and return the configured FastAPI application."""
    context = _make_context(config, strategy)
    manifest = context.store.read_manifest()

    app = FastAPI(title="MergenVision Video Reference Annotator")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/config")
    async def get_config() -> dict[str, Any]:
        return {
            "run_dir": str(context.run_dir),
            "strategy": context.strategy,
            "video_path": str(context.video_path),
            "video_sha256": manifest.video_sha256,
            "display_width": manifest.display_width,
            "display_height": manifest.display_height,
            "duration_ns": manifest.duration_ns,
            "decoded_frame_count": manifest.decoded_frame_count,
            "sampled_frame_count": manifest.sampled_frame_count,
            "processed_frame_count": manifest.processed_frame_count,
            "overlay_path": str(context.overlay_path),
            "gt_path": str(context.gt_path),
        }

    @app.get("/api/frames")
    async def get_frames() -> dict[str, Any]:
        frames = context.store.read_frames()
        return {
            "count": len(frames),
            "scene_cut_indices": manifest.scene_cut_frame_indices,
            "frames": [
                {
                    "frame_index": f.frame_index,
                    "pts_ns": f.pts_ns,
                    "sampled": f.sampled,
                    "processed": f.processed,
                }
                for f in frames
            ],
        }

    @app.get("/api/overlay")
    async def get_overlay() -> dict[str, Any]:
        if not context.overlay_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"overlay not found: {context.overlay_path}. Run visualize first.",
            )
        return {"by_frame": _load_overlay(context.overlay_path)}

    @app.get("/api/tracklets")
    async def get_tracklets() -> list[dict[str, Any]]:
        if not context.tracklets_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"tracklets not found: {context.tracklets_path}. Run replay first.",
            )
        return _load_jsonl(context.tracklets_path)

    @app.get("/api/canonical")
    async def get_canonical() -> list[dict[str, Any]]:
        return _load_canonical_tracks(context.canonical_path)

    @app.get("/api/video")
    async def get_video(request: Request) -> StreamingResponse:
        return _range_response(context.video_path, request)

    @app.head("/api/video")
    async def head_video(request: Request) -> StreamingResponse:
        response = _range_response(context.video_path, request)
        response.body = b""
        return response

    @app.get("/api/gt")
    async def get_gt() -> GroundTruth:
        return _load_gt(context)

    @app.post("/api/gt")
    async def post_gt(save: AnchorSave) -> dict[str, Any]:
        gt = _load_gt(context)
        gt.anchors = save.anchors
        _save_gt(context.gt_path, gt)
        return {"saved": True, "anchor_count": len(gt.anchors)}

    @app.get("/api/gt/resolve")
    async def resolve_gt() -> dict[str, Any]:
        gt = _load_gt(context)
        observations = context.store.read_observations()
        resolved = resolve_anchor_observations(gt, observations)
        return {
            "anchor_count": len(gt.anchors),
            "resolved_count": sum(1 for r in resolved.values() if r["resolved"]),
            "anchors": resolved,
        }

    return app

#!/usr/bin/env python3
"""Benchmark LFW enrollment through the existing /api/v1/faces API.

Uses 3 backend containers (one per GPU) and keeps each GPU busy with bounded
concurrent HTTP requests.  For each LFW subject:

1. POST /api/v1/faces/recognize  (first image)
2. POST /api/v1/faces/{faceId}/enroll
3. POST /api/v1/faces/{faceId}/samples  (remaining images)

All persistence goes through the backend into PostgreSQL/MinIO/Qdrant.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx


API_BASES = ["http://localhost:8090", "http://localhost:8091", "http://localhost:8092"]


@dataclass
class Subject:
    name: str
    images: list[Path]


@dataclass
class Stats:
    subjects_total: int = 0
    subjects_enrolled: int = 0
    subjects_failed: int = 0
    images_total: int = 0
    images_enrolled: int = 0
    images_sample_added: int = 0
    images_failed: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subjects": {
                "total": self.subjects_total,
                "enrolled": self.subjects_enrolled,
                "failed": self.subjects_failed,
            },
            "images": {
                "total": self.images_total,
                "first_enrolled": self.images_enrolled,
                "samples_added": self.images_sample_added,
                "failed": self.images_failed,
            },
            "timing": self.timing,
            "errors_tail": self.errors[-20:],
        }


def load_subjects(dataset_root: Path) -> list[Subject]:
    subjects: list[Subject] = []
    for entry in sorted(Path(dataset_root).iterdir()):
        if not entry.is_dir():
            continue
        images = sorted(p for p in entry.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"})
        if not images:
            continue
        subjects.append(Subject(name=entry.name, images=images))
    return subjects


class BackendPool:
    def __init__(self, bases: list[str], concurrency_per_backend: int, timeout: float = 120.0) -> None:
        self.bases = bases
        self.clients = [httpx.AsyncClient(timeout=timeout) for _ in bases]
        self.sems = [asyncio.Semaphore(concurrency_per_backend) for _ in bases]
        self._counter = itertools.cycle(range(len(bases)))

    async def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        idx = next(self._counter)
        async with self.sems[idx]:
            return await self.clients[idx].request(method, f"{self.bases[idx]}{path}", **kwargs)

    async def close(self) -> None:
        await asyncio.gather(*(c.aclose() for c in self.clients))


async def recognize(pool: BackendPool, image_path: Path) -> tuple[dict[str, Any] | None, str]:
    with image_path.open("rb") as f:
        data = f.read()
    resp = await pool.request("POST", "/api/v1/faces/recognize", files={"image": (image_path.name, data, "image/jpeg")})
    if resp.status_code != 200:
        return None, f"status={resp.status_code} body={resp.text[:200]}"
    return resp.json(), ""


async def enroll_known(pool: BackendPool, face_id: str, name: str) -> bool:
    resp = await pool.request("POST", f"/api/v1/faces/{face_id}/enroll", json={"name": name})
    return resp.status_code == 200


async def add_sample(pool: BackendPool, face_id: str, image_path: Path) -> bool:
    with image_path.open("rb") as f:
        data = f.read()
    resp = await pool.request("POST", f"/api/v1/faces/{face_id}/samples", files={"image": (image_path.name, data, "image/jpeg")})
    return resp.status_code == 200


async def main() -> None:
    parser = argparse.ArgumentParser(description="LFW enrollment API benchmark")
    parser.add_argument("--dataset-root", default="phase1/gpu_bulk_enrollment/lfw", type=Path)
    parser.add_argument("--concurrency-per-backend", type=int, default=4)
    parser.add_argument("--subject-concurrency", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="only process first N subjects (smoke test)")
    parser.add_argument("--progress-every", type=int, default=100, help="log progress every N subjects")
    parser.add_argument("--progress-interval", type=float, default=10.0, help="force progress log every N seconds")
    parser.add_argument("--output", default="backend/scripts/lfw_benchmark_result.json", type=Path)
    args = parser.parse_args()

    subjects = load_subjects(args.dataset_root)
    if args.limit is not None:
        subjects = subjects[:args.limit]

    stats = Stats(
        subjects_total=len(subjects),
        images_total=sum(len(s.images) for s in subjects),
    )

    progress_lock = asyncio.Lock()
    last_progress = time.perf_counter()
    processed_subjects = 0

    def log_progress(force: bool = False) -> None:
        nonlocal last_progress
        now = time.perf_counter()
        elapsed = now - start
        if not force and elapsed - last_progress < args.progress_interval and processed_subjects % args.progress_every != 0:
            return
        last_progress = now
        imgs_done = stats.images_enrolled + stats.images_sample_added + stats.images_failed
        print(
            f"[{elapsed:7.1f}s] subjects={processed_subjects}/{stats.subjects_total} "
            f"images={imgs_done}/{stats.images_total} "
            f"fps={imgs_done / elapsed:6.2f} "
            f"enrolled_subjects={stats.subjects_enrolled} "
            f"failed_images={stats.images_failed} "
            f"errors={len(stats.errors)}",
            flush=True,
        )

    pool = BackendPool(API_BASES, concurrency_per_backend=args.concurrency_per_backend)
    subject_sem = asyncio.Semaphore(args.subject_concurrency)

    async def process_subject(subject: Subject) -> None:
        nonlocal processed_subjects
        async with subject_sem:
            first = subject.images[0]
            try:
                rec, err = await recognize(pool, first)
            except Exception as exc:
                stats.images_failed += len(subject.images)
                stats.subjects_failed += 1
                stats.errors.append({"subject": subject.name, "stage": "recognize", "error": str(exc)})
                return
            finally:
                async with progress_lock:
                    processed_subjects += 1
                    log_progress()

            if rec is None:
                stats.images_failed += len(subject.images)
                stats.subjects_failed += 1
                stats.errors.append({"subject": subject.name, "stage": "recognize", "error": err})
                return
            if rec.get("faceCount", 0) == 0:
                stats.images_failed += len(subject.images)
                stats.subjects_failed += 1
                stats.errors.append({"subject": subject.name, "stage": "recognize", "error": "no_face"})
                return

            face_id = rec["faces"][0]["faceId"]
            try:
                enrolled = await enroll_known(pool, face_id, subject.name)
            except Exception as exc:
                stats.images_failed += len(subject.images)
                stats.subjects_failed += 1
                stats.errors.append({"subject": subject.name, "stage": "enroll", "error": str(exc)})
                return

            if enrolled:
                stats.subjects_enrolled += 1
                stats.images_enrolled += 1
            elif rec["faces"][0].get("status") in ("known", "anonymous"):
                # Identity already exists from a prior run; reuse it for samples.
                pass
            else:
                stats.images_failed += len(subject.images)
                stats.subjects_failed += 1
                stats.errors.append({"subject": subject.name, "stage": "enroll", "error": "http_error"})
                return

            if len(subject.images) == 1:
                return

            sample_tasks = [add_sample(pool, face_id, img) for img in subject.images[1:]]
            results = await asyncio.gather(*sample_tasks, return_exceptions=True)
            for ok in results:
                if ok is True:
                    stats.images_sample_added += 1
                else:
                    stats.images_failed += 1
                    if isinstance(ok, Exception):
                        stats.errors.append({"subject": subject.name, "stage": "sample", "error": str(ok)})

    start = time.perf_counter()
    tasks = [process_subject(s) for s in subjects]
    await asyncio.gather(*tasks)
    log_progress(force=True)
    elapsed = time.perf_counter() - start

    await pool.close()

    stats.timing = {
        "wall_seconds": elapsed,
        "images_per_second": stats.images_total / elapsed if elapsed > 0 else 0.0,
        "subjects_per_second": len(subjects) / elapsed if elapsed > 0 else 0.0,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(stats.to_dict(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())

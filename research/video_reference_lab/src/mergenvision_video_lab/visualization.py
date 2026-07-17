"""Visual diagnostics: contact sheets, histograms, timeline, debug MP4, HTML report."""

from __future__ import annotations

import base64
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from mergenvision_video_lab.alignment import align_face
from mergenvision_video_lab.contracts import FaceObservation
from mergenvision_video_lab.video_reader import VideoReader


def _color_for_id(id_str: str) -> tuple[int, int, int]:
    """Deterministic BGR color from an ID string."""
    h = hash(id_str) & 0xFFFFFFFF
    r = (h >> 16) & 0xFF
    g = (h >> 8) & 0xFF
    b = h & 0xFF
    return (int(b), int(g), int(r))


def _draw_observation(
    img: np.ndarray,
    obs: FaceObservation,
    label: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = obs.bbox_xyxy.to_list()
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        img,
        label,
        (x1, max(y1 - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )
    for _name, (px, py) in zip(
        ["LE", "RE", "N", "LM", "RM"],
        obs.landmarks_5.to_array(),
        strict=False,
    ):
        cv2.circle(img, (int(px), int(py)), 2, (0, 255, 0), -1)
    return img


def _crop_face(img: np.ndarray, obs: FaceObservation, pad: float = 0.5) -> np.ndarray:
    x1, y1, x2, y2 = obs.bbox_xyxy.to_list()
    w = x2 - x1
    h = y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    size = max(w, h) * (1 + pad)
    x1c = max(0, int(cx - size / 2))
    y1c = max(0, int(cy - size / 2))
    x2c = min(img.shape[1], int(cx + size / 2))
    y2c = min(img.shape[0], int(cy + size / 2))
    crop = img[y1c:y2c, x1c:x2c]
    return crop


def _frames_by_index(video_path: Path, frame_indices: set[int]) -> dict[int, np.ndarray]:
    """Load a sparse set of original-resolution frames."""
    result: dict[int, np.ndarray] = {}
    reader = VideoReader(video_path)
    for frame in reader:
        if frame.frame_index in frame_indices:
            result[frame.frame_index] = frame.ndarray.copy()
        if len(result) == len(frame_indices):
            break
    return result


def make_alignment_contact_sheet(
    video_path: Path,
    observations: list[FaceObservation],
    output_path: Path,
    alignment_config: dict[str, Any] | None = None,
    columns: int = 4,
    max_samples: int = 16,
) -> None:
    """Contact sheet of original crop + real aligned crop + metrics.

    The aligned crop is produced by the same ``align_face`` function used by the
    recognition oracle, so this sheet verifies landmark order, similarity
    transform, color order and reprojection error in one glance.
    """
    cfg = alignment_config or {}
    eligible = [obs for obs in observations if obs.recognition_eligible]
    if not eligible:
        return
    samples = eligible[:max_samples]
    frame_indices = {obs.frame_index for obs in samples}
    frames = _frames_by_index(video_path, frame_indices)

    rows = (len(samples) + columns - 1) // columns
    cell_w, cell_h = 280, 180
    sheet = Image.new("RGB", (cell_w * columns, cell_h * rows), (40, 40, 40))

    output_size = int(cfg.get("output_size", 112))
    color_order = cfg.get("color_order", "BGR")
    border_mode = cfg.get("border_mode", "constant_zero")
    interpolation = cfg.get("interpolation", "bilinear")

    for idx, obs in enumerate(samples):
        frame = frames.get(obs.frame_index)
        if frame is None:
            continue

        original = _crop_face(frame, obs)
        original = cv2.resize(original, (112, 112))

        aligned: np.ndarray | None = None
        reproj: float | None = None
        try:
            aligned, _matrix, reproj = align_face(
                frame,
                obs.landmarks_5.to_array(),
                output_size=output_size,
                color_order=color_order,
                border_mode=border_mode,
                interpolation=interpolation,
            )
        except Exception:
            aligned = None
            reproj = None

        cell = Image.new("RGB", (cell_w, cell_h), (40, 40, 40))
        cell.paste(Image.fromarray(original), (8, 8))
        if aligned is not None:
            aligned_preview = cv2.resize(aligned, (112, 112))
            cell.paste(Image.fromarray(aligned_preview), (140, 8))

        label = (
            f"{obs.observation_id}\n"
            f"f:{obs.frame_index} pts:{obs.pts_ns}\n"
            f"det:{obs.detector_score:.2f} q:{obs.quality.composite_quality_score:.2f}\n"
            f"lap:{obs.quality.grayscale_laplacian_variance:.0f}"
        )
        if reproj is not None:
            label += f" reproj:{reproj:.2f}px"
        else:
            label += " align:FAIL"

        cell_arr = np.array(cell)
        cv2.putText(
            cell_arr,
            label,
            (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        sheet.paste(
            Image.fromarray(cell_arr), ((idx % columns) * cell_w, (idx // columns) * cell_h)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)


def _tracklet_id(t: Any) -> str:
    return str(t.raw_tracklet_id if hasattr(t, "raw_tracklet_id") else t["raw_tracklet_id"])


def _tracklet_obs_ids(t: Any) -> list[str]:
    ids: Any
    if hasattr(t, "observation_ids"):
        ids = t.observation_ids
    elif hasattr(t, "detection_ordinal_ids"):
        ids = t.detection_ordinal_ids
    else:
        ids = t.get("observation_ids") or t.get("detection_ordinal_ids") or []
    return [str(x) for x in ids]


def _tracklet_pts(t: Any) -> tuple[int, int]:
    first = int(t.first_pts_ns if hasattr(t, "first_pts_ns") else t["first_pts_ns"])
    last = int(t.last_pts_ns if hasattr(t, "last_pts_ns") else t["last_pts_ns"])
    return first, last


def make_tracklet_contact_sheet(
    video_path: Path,
    tracklets: list[Any],
    observations: list[FaceObservation],
    output_path: Path,
    columns: int = 4,
) -> None:
    """One row per raw tracklet: first/highest/middle/last accepted observation."""
    obs_by_id = {obs.observation_id: obs for obs in observations}
    rows = []
    frame_indices: set[int] = set()
    for tracklet in tracklets:
        obs_ids = _tracklet_obs_ids(tracklet)
        if not obs_ids:
            continue
        accepted = [obs_by_id[oid] for oid in obs_ids if oid in obs_by_id]
        if not accepted:
            continue
        accepted.sort(key=lambda o: o.pts_ns)
        chosen = [
            accepted[0],
            max(accepted, key=lambda o: o.quality.composite_quality_score),
            accepted[len(accepted) // 2],
            accepted[-1],
        ]
        rows.append((tracklet, chosen))
        frame_indices.update(o.frame_index for o in chosen)

    if not rows:
        return
    frames = _frames_by_index(video_path, frame_indices)
    cell_size = 112
    sheet = Image.new("RGB", (cell_size * columns, cell_size * len(rows)), (40, 40, 40))
    for r, (tracklet, chosen) in enumerate(rows):
        for c, obs in enumerate(chosen[:columns]):
            frame = frames.get(obs.frame_index)
            if frame is None:
                continue
            crop = _crop_face(frame, obs, pad=0.3)
            crop = cv2.resize(crop, (cell_size, cell_size))
            label = f"{_tracklet_id(tracklet)} q:{obs.quality.composite_quality_score:.2f}"
            cv2.putText(crop, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            sheet.paste(Image.fromarray(crop), (c * cell_size, r * cell_size))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)


def make_canonical_contact_sheet(
    video_path: Path,
    canonical_tracks: list[Any],
    observations: list[FaceObservation],
    output_path: Path,
    columns: int = 4,
) -> None:
    """One row per canonical track: first/highest/middle/last detection."""
    obs_by_id = {obs.observation_id: obs for obs in observations}
    rows = []
    frame_indices: set[int] = set()
    for track in canonical_tracks:
        detections = track.detections
        if not detections:
            continue
        det_ids = [d["observation_id"] for d in detections]
        accepted = [obs_by_id[oid] for oid in det_ids if oid in obs_by_id]
        if not accepted:
            continue
        accepted.sort(key=lambda o: o.pts_ns)
        chosen = [
            accepted[0],
            max(accepted, key=lambda o: o.quality.composite_quality_score),
            accepted[len(accepted) // 2],
            accepted[-1],
        ]
        rows.append((track, chosen))
        frame_indices.update(o.frame_index for o in chosen)

    if not rows:
        return
    frames = _frames_by_index(video_path, frame_indices)
    cell_size = 112
    sheet = Image.new("RGB", (cell_size * columns, cell_size * len(rows)), (40, 40, 40))
    for r, (track, chosen) in enumerate(rows):
        for c, obs in enumerate(chosen[:columns]):
            frame = frames.get(obs.frame_index)
            if frame is None:
                continue
            crop = _crop_face(frame, obs, pad=0.3)
            crop = cv2.resize(crop, (cell_size, cell_size))
            label = f"{track.canonical_track_id}"
            cv2.putText(crop, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            sheet.paste(Image.fromarray(crop), (c * cell_size, r * cell_size))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)


def make_quality_histograms(
    observations: list[FaceObservation],
    output_path: Path,
) -> None:
    """Histograms of raw quality metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    composites = [obs.quality.composite_quality_score for obs in observations]
    detector_scores = [obs.detector_score for obs in observations]
    lap_vars = [obs.quality.grayscale_laplacian_variance for obs in observations]
    sizes = [obs.quality.bbox_min_side_px for obs in observations]

    for ax, values, title in zip(
        axes,
        [composites, detector_scores, lap_vars, sizes],
        ["Composite quality", "Detector score", "Laplacian variance", "Min bbox side (px)"],
        strict=False,
    ):
        ax.hist(values, bins=40, color="steelblue", edgecolor="black")
        ax.set_title(title)
        ax.set_ylabel("Count")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def make_cosine_histograms(
    diagnostics: dict[str, Any],
    output_path: Path,
) -> None:
    """Histograms of cosine similarity distributions."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, values in diagnostics.items():
        if isinstance(values, list) and values:
            ax.hist(values, bins=40, alpha=0.6, label=label)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Count")
    ax.set_title("Cosine diagnostics")
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def make_timeline(
    tracklets: list[Any],
    canonical_map: dict[str, str],
    output_path: Path,
) -> None:
    """Timeline plot of raw tracklets colored by canonical track.

    Accepts either live Tracklet/RawTrackletSummary objects or plain dicts
    produced by ``replay_frames`` serialization.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    y_positions: dict[str, int] = {}
    for tracklet in sorted(tracklets, key=lambda t: (_tracklet_pts(t)[0], _tracklet_id(t))):
        y_positions[_tracklet_id(tracklet)] = len(y_positions)

    for tracklet in tracklets:
        tid = _tracklet_id(tracklet)
        y = y_positions[tid]
        ct_id = canonical_map.get(tid, "unresolved")
        channels = _color_for_id(ct_id)
        color = (channels[0] / 255.0, channels[1] / 255.0, channels[2] / 255.0)
        first_pts_ns, last_pts_ns = _tracklet_pts(tracklet)
        ax.hlines(
            y,
            first_pts_ns / 1e9,
            last_pts_ns / 1e9,
            colors=[color],
            linewidths=3,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Raw tracklet")
    ax.set_title("Raw tracklet timeline (colored by canonical track)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def make_overlay_jsonl(
    observations: list[FaceObservation],
    assignments: list[dict[str, Any]],
    canonical_map: dict[str, str],
    labels: dict[str, str | None],
    output_path: Path,
) -> None:
    """Write per-frame overlay metadata."""
    obs_by_id = {obs.observation_id: obs for obs in observations}
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for assignment in assignments:
        obs = obs_by_id.get(assignment["observation_id"])
        if obs is None:
            continue
        rt_id = assignment["raw_tracklet_id"]
        ct_id = canonical_map.get(rt_id, "unresolved")
        label = labels.get(ct_id)
        by_frame[obs.frame_index].append(
            {
                "observation_id": obs.observation_id,
                "pts_ns": obs.pts_ns,
                "bbox_xyxy": obs.bbox_xyxy.to_list(),
                "raw_tracklet_id": rt_id,
                "canonical_track_id": ct_id,
                "display_label": label,
                "detector_score": obs.detector_score,
                "quality_score": obs.quality.composite_quality_score,
                "tracking_eligible": obs.tracking_eligible,
                "recognition_eligible": obs.recognition_eligible,
                "rejection_reasons": obs.rejection_reasons.copy(),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for frame_index in sorted(by_frame):
            record = {"frame_index": frame_index, "faces": by_frame[frame_index]}
            f.write(json.dumps(record) + "\n")


def render_debug_mp4(
    video_path: Path,
    observations: list[FaceObservation],
    assignments: list[dict[str, Any]],
    canonical_map: dict[str, str],
    labels: dict[str, str | None],
    output_path: Path,
    preserve_audio: bool = True,
) -> dict[str, Any]:
    """Render a debug annotated MP4 from the original video and overlay metadata."""
    obs_by_id = {obs.observation_id: obs for obs in observations}
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for assignment in assignments:
        obs = obs_by_id.get(assignment["observation_id"])
        if obs is None:
            continue
        by_frame[obs.frame_index].append(
            {
                "obs": obs,
                "raw_tracklet_id": assignment["raw_tracklet_id"],
                "canonical_track_id": canonical_map.get(
                    assignment["raw_tracklet_id"], "unresolved"
                ),
                "label": labels.get(canonical_map.get(assignment["raw_tracklet_id"], "")),
            }
        )

    reader = VideoReader(video_path)
    probe = reader.probe
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        float(probe.avg_frame_rate) or 30.0,
        (probe.display_width, probe.display_height),
    )

    source_frame_count = 0
    output_frame_count = 0
    for frame in reader:
        source_frame_count += 1
        img = frame.ndarray.copy()
        for ann in by_frame.get(frame.frame_index, []):
            obs = ann["obs"]
            label = ann["label"] or "unresolved"
            text = (
                f"{ann['raw_tracklet_id']} | {ann['canonical_track_id']} | {label} | "
                f"det:{obs.detector_score:.2f} | q:{obs.quality.composite_quality_score:.2f}"
            )
            color = _color_for_id(ann["canonical_track_id"])
            _draw_observation(img, obs, text, color)
        writer.write(img)
        output_frame_count += 1

    writer.release()
    return {
        "source_frame_count": source_frame_count,
        "output_frame_count": output_frame_count,
        "audio_preserved": False,
        "preserve_audio_requested": preserve_audio,
        "note": "audio not preserved by OpenCV VideoWriter",
    }


def make_html_report(
    report_data: dict[str, Any],
    output_path: Path,
) -> None:
    """Write a minimal HTML report embedding summary cards."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>MergenVision Video Reference Lab Report</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; background: #f8f9fa; }}
.card {{ background: white; padding: 1rem; margin-bottom: 1rem; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
pre {{ background: #f1f3f4; padding: 0.75rem; overflow-x: auto; }}
h1, h2 {{ color: #202124; }}
table {{ border-collapse: collapse; margin-top: 0.5rem; }}
th, td {{ border: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: left; }}
</style>
</head>
<body>
<h1>MergenVision Video Reference Lab Report</h1>
<div class="card">
<h2>Verdict</h2>
<pre>{report_data.get("verdict", "UNKNOWN")}</pre>
</div>
<div class="card">
<h2>Manifest Summary</h2>
<pre>{json.dumps(report_data.get("manifest", {}), indent=2)}</pre>
</div>
<div class="card">
<h2>Tracking</h2>
<pre>{json.dumps(report_data.get("tracking", {}), indent=2)}</pre>
</div>
<div class="card">
<h2>Reconciliation</h2>
<pre>{json.dumps(report_data.get("reconciliation", {}), indent=2)}</pre>
</div>
<div class="card">
<h2>Benchmark</h2>
<pre>{json.dumps(report_data.get("benchmark", {}), indent=2)}</pre>
</div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def make_reference_lab_html(
    report_data: dict[str, Any],
    visual_dir: Path,
    output_path: Path,
) -> None:
    """Build a richer HTML report with embedded contact-sheet thumbnails."""
    images = [p for p in visual_dir.iterdir() if p.suffix.lower() in (".jpg", ".png")]
    image_cards = []
    for img_path in sorted(images):
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = img_path.suffix.lstrip(".")
        image_cards.append(
            f'<div class="card"><h3>{img_path.name}</h3>'
            f'<img src="data:image/{ext};base64,{b64}" style="max-width:100%;"></div>'
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Video Reference Lab — Review Package</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; background: #f8f9fa; }}
.card {{ background: white; padding: 1rem; margin-bottom: 1rem; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
pre {{ background: #f1f3f4; padding: 0.75rem; overflow-x: auto; }}
h1, h2 {{ color: #202124; }}
</style>
</head>
<body>
<h1>Video Reference Lab Review Package</h1>
<div class="card">
<h2>Final Report</h2>
<pre>{json.dumps(report_data, indent=2)}</pre>
</div>
{"".join(image_cards)}
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")

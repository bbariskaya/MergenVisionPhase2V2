"""End-to-end smoke test on the Friends reference video.

BLOCKED unless ``Friends.mp4`` exists and models are present.
When blocked the test is skipped cleanly with a message so CI remains green.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from mergenvision_video_lab.config import load_config
from mergenvision_video_lab.pipeline import (
    stage_extract,
    stage_gallery,
    stage_reconcile,
    stage_replay,
    stage_templates,
    stage_validate,
)

FRIENDS_PATH = Path("/home/user/Workspace/MergenVisionPhase2v2/test_videos/Friends.mp4")


def _check_friends_blocked() -> str | None:
    """Return a skip reason string if the Friends smoke cannot proceed."""
    if not FRIENDS_PATH.exists():
        return "Friends.mp4 not found"
    if shutil.which("ffprobe") is None:
        return "ffprobe not installed"
    config = load_config("configs/friends_baseline_cpu.yaml")
    if not config.oracle.model_pack:
        return "no model pack configured"
    try:
        from mergenvision_video_lab.oracle.insightface_oracle import InsightFaceOracle

        _ = InsightFaceOracle(
            model_root=config.oracle.model_root,
            model_pack=config.oracle.model_pack,
            requested_provider=config.oracle.provider,
            allow_cpu_fallback=config.oracle.allow_cpu_fallback,
            det_size=tuple(config.oracle.det_size),
        )
    except Exception as exc:
        return f"model not loadable: {exc}"
    return None


@pytest.mark.friends
@pytest.mark.skipif(
    bool(reason := _check_friends_blocked()),
    reason=reason or "unknown block",
)
def test_friends_32_frame_pipeline(tmp_path: Path) -> None:
    """Run the first 32 frames of Friends through extraction, replay, templates,
    reconciliation, gallery and validation stages.
    """
    config = load_config("configs/friends_baseline_cpu.yaml")
    # Force a short, deterministic run inside the temp directory.
    config.output.base_dir = str(tmp_path / "artifacts")
    config.video.path = str(FRIENDS_PATH)
    config.video.max_frames = 32

    extract_result = stage_extract(config)
    run_dir = Path(extract_result["run_dir"])
    assert run_dir.exists()

    replay_result = stage_replay(config, strategy="byte_iou")
    assert (run_dir / "replay" / "byte_iou" / "assignments.jsonl").exists()
    assert replay_result["assignment_count"] >= 0

    stage_templates(config, strategy="byte_iou")
    assert (run_dir / "tracks" / "tracklet_templates.jsonl").exists()

    stage_reconcile(config, strategy="byte_iou")
    assert (run_dir / "tracks" / "canonical_tracks.jsonl").exists()

    gallery_result = stage_gallery(config, strategy="byte_iou")
    if gallery_result is not None:
        assert (run_dir / "gallery" / "decisions.json").exists()

    summary = stage_validate(config)
    assert summary["valid"]
    assert summary["manifest"]["decoded_frame_count"] > 0

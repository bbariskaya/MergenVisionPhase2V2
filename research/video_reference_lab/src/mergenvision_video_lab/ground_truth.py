"""Minimal human-label checkpoint loading and anchor resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mergenvision_video_lab.contracts import GroundTruth, GroundTruthAnchor
from mergenvision_video_lab.errors import ConfigError


def load_ground_truth(path: Path | str) -> GroundTruth:
    """Load a ground-truth YAML file."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"ground truth file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError("ground truth file must contain a YAML mapping")
    return GroundTruth(**raw)


def build_ground_truth_template(video_sha256: str) -> GroundTruth:
    """Return the minimal human-label checkpoint template."""
    return GroundTruth(
        video_sha256=video_sha256,
        anchors=[
            GroundTruthAnchor(
                anchor_id="rachel_early",
                label="Rachel",
                split="calibration",
            ),
            GroundTruthAnchor(
                anchor_id="rachel_late",
                label="Rachel",
                split="holdout",
            ),
        ],
    )


def resolve_anchor_observations(
    ground_truth: GroundTruth,
    observations: list[Any],
) -> dict[str, Any]:
    """Resolve ground-truth anchors to actual observation records."""
    by_id = {obs.observation_id: obs for obs in observations}
    resolved: dict[str, Any] = {}
    for anchor in ground_truth.anchors:
        obs = None
        if anchor.observation_id:
            obs = by_id.get(anchor.observation_id)
        elif anchor.frame_index is not None:
            matches = [o for o in observations if o.frame_index == anchor.frame_index]
            if matches:
                # Prefer highest-quality face if multiple.
                obs = max(matches, key=lambda o: o.quality.composite_quality_score)
        resolved[anchor.anchor_id] = {
            "anchor": anchor,
            "observation": obs,
            "resolved": obs is not None,
        }
    return resolved

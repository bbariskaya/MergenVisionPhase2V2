# MergenVision Video Reference Lab

Isolated offline reference/correctness laboratory for video face detection, tracking, and recognition.

This is **not** the production pipeline. It exists to:

1. Freeze frame/detection/landmark/quality/embedding observations from a source video once.
2. Replay tracking and reconciliation without decoding or running inference again.
3. Prove chunk-invariance of tracker state.
4. Separate `observation_id`, `raw_tracklet_id`, `canonical_track_id`, and `display_label` concepts.
5. Produce visual diagnostics and a human-label checkpoint instead of fabricating identities.

## Quick start

```bash
# Install (CPU path)
make video-reference-install

# Verify environment and models
make video-reference-doctor

# Run unit/synthetic tests
make video-reference-unit

# Run bounded smoke extraction on Friends.mp4
make video-reference-smoke

# Run the full lab
make video-reference-friends
```

## Layout

- `src/mergenvision_video_lab/` — implementation.
- `tests/` — unit and integration tests.
- `configs/friends_baseline.yaml` — thresholds and algorithm choices.
- `configs/friends_ground_truth.template.yaml` — template for human labels.
- `notebooks/friends_reference_lab.ipynb` — thin visualization client.

## Artifacts

All generated artifacts live under `artifacts/video_reference/<video_sha12>/<config_sha12>/`.

Do not commit artifacts, models, videos, or gallery images.

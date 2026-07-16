"""ByteTrack lifecycle with appearance-aware first-stage association.

The face-specific modification is described in the Sprint 002 spec:
when both a track and a high-score detection have a valid embedding, the
first-stage Hungarian cost is a weighted sum of IoU distance and appearance
(1 - cosine) distance.  The second low-score stage remains geometry-only so
unreliable embeddings cannot poison identity association.

Base lifecycle is adapted from FoundationVision/ByteTrack
(https://github.com/FoundationVision/ByteTrack), yolox/tracker/byte_tracker.py,
commit d1bf0191adff59bc8fcfeaa0b33d3d1642552a99, MIT license.
"""

from __future__ import annotations

import numpy as np

from mergenvision_video_lab.geometry import iou_xyxy
from mergenvision_video_lab.tracking.assignment import _GATED_COST, linear_assignment
from mergenvision_video_lab.tracking.byte_tracker import ByteTrackIoUTracker, Detection, Tracklet


class HybridFaceByteTracker(ByteTrackIoUTracker):
    """ByteTrack with a face-embedding fused first association stage."""

    strategy = "hybrid"

    def _first_stage_association(
        self,
        strack_pool: list[Tracklet],
        dets_high: list[Detection],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Fuse IoU and appearance for activated/lost tracks vs high detections."""
        if not strack_pool or not dets_high:
            return [], list(range(len(strack_pool))), list(range(len(dets_high)))

        min_iou = float(self.config["first_stage_min_iou"])
        min_cosine = float(self.config["short_term_min_cosine"])
        appearance_weight = float(self.config["appearance_weight"])
        iou_weight = 1.0 - appearance_weight

        cost = np.zeros((len(strack_pool), len(dets_high)), dtype=np.float64)
        for i, track in enumerate(strack_pool):
            template = track.appearance_template
            for j, det in enumerate(dets_high):
                iou = iou_xyxy(track.tlbr, det.tlbr)
                if iou < min_iou:
                    cost[i, j] = _GATED_COST
                    continue

                iou_distance = 1.0 - iou
                if template is not None and det.embedding is not None:
                    cosine = float(np.dot(template, det.embedding))
                    if cosine < min_cosine:
                        cost[i, j] = _GATED_COST
                        continue
                    appearance_distance = 1.0 - cosine
                    cost[i, j] = iou_weight * iou_distance + appearance_weight * appearance_distance
                else:
                    cost[i, j] = iou_distance

        return linear_assignment(cost, thresh=1.0)

    def _after_track_update(self, track: Tracklet, det: Detection) -> None:
        """Maintain the short-term appearance template after every update."""
        if det.embedding is None:
            return
        track.update_appearance(
            det.embedding,
            det.quality,
            det.pts_ns,
            top_k=int(self.config["evidence_top_k"]),
            min_separation_ns=int(self.config["evidence_min_separation_ns"]),
        )

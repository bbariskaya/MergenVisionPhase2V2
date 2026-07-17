from app.infrastructure.serialization.video_observation_v1_pb2 import (
    FaceDetection,
    ObservationChunkFooter,
    VideoObservationFrame,
)
from app.infrastructure.serialization.video_track_template_v1_pb2 import (
    RawTrackTemplate,
    TrackTemplateBundle,
    TrackTemplateFooter,
)

__all__ = [
    "FaceDetection",
    "ObservationChunkFooter",
    "VideoObservationFrame",
    "RawTrackTemplate",
    "TrackTemplateBundle",
    "TrackTemplateFooter",
]

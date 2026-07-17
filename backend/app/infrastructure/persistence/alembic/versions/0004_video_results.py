"""Video result schema (tracks, tracklets, appearances, timeline).

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-17 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_TRACK_STATUS_ALLOWED = (
    "known",
    "anonymous",
    "new_anonymous",
)
_TRACKLET_STATE_ALLOWED = (
    "confirmed",
    "lost",
    "removed",
)
_ARTIFACT_KIND_ALLOWED = (
    "private_observation",
    "public_overlay",
)
_SAMPLE_PURPOSE_ALLOWED = (
    "identity_seed",
    "best_shot",
    "gallery_admission",
)


def _empty_jsonb() -> sa.schema.DefaultClause | sa.TextClause:
    return sa.text("'{}'::jsonb")


def upgrade() -> None:
    # ------------------------------------------------------------------
    # Extend video_job with result manifest fields
    # ------------------------------------------------------------------
    op.add_column("video_job", sa.Column("result_manifest_bucket", sa.String(255), nullable=True))
    op.add_column("video_job", sa.Column("result_manifest_key", sa.String(1024), nullable=True))
    op.add_column("video_job", sa.Column("result_manifest_sha256", sa.CHAR(64), nullable=True))
    op.add_column("video_job", sa.Column("result_schema_version", sa.String(32), nullable=True))

    # ------------------------------------------------------------------
    # video_track
    # ------------------------------------------------------------------
    op.create_table(
        "video_track",
        sa.Column("track_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("video_job.job_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("track_ordinal", sa.Integer, nullable=False),
        sa.Column(
            "face_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("face_identity.face_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "recognition_result_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("recognition_result.result_id", ondelete="RESTRICT"),
            nullable=False,
            unique=True,
        ),
        sa.Column("status_at_processing", sa.String(16), nullable=False),
        sa.Column("name_at_processing", sa.String(255), nullable=True),
        sa.Column("metadata_at_processing", postgresql.JSONB, nullable=False, server_default=_empty_jsonb()),
        sa.Column("identity_version_at_processing", sa.Integer, nullable=False),
        sa.Column("match_confidence", sa.REAL, nullable=False),
        sa.Column("top1_score", sa.REAL, nullable=True),
        sa.Column("top2_score", sa.REAL, nullable=True),
        sa.Column("margin_score", sa.REAL, nullable=True),
        sa.Column("threshold_used", sa.REAL, nullable=True),
        sa.Column("first_frame_index", sa.BigInteger, nullable=False),
        sa.Column("last_frame_index", sa.BigInteger, nullable=False),
        sa.Column("first_pts_ns", sa.BigInteger, nullable=False),
        sa.Column("last_pts_ns", sa.BigInteger, nullable=False),
        sa.Column("total_duration_ns", sa.BigInteger, nullable=False),
        sa.Column("detection_count", sa.BigInteger, nullable=False),
        sa.Column("tracklet_count", sa.Integer, nullable=False),
        sa.Column(
            "best_sample_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("face_sample.sample_id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("job_id", "track_ordinal", name="uq_video_track_job_ordinal"),
        sa.CheckConstraint(
            sa.text(f"status_at_processing IN {str(_TRACK_STATUS_ALLOWED)}"),
            name="ck_video_track_status",
        ),
        sa.CheckConstraint(
            sa.text("first_frame_index <= last_frame_index"),
            name="ck_video_track_frame_order",
        ),
        sa.CheckConstraint(
            sa.text("first_pts_ns <= last_pts_ns"),
            name="ck_video_track_pts_order",
        ),
        sa.CheckConstraint(
            sa.text("total_duration_ns >= 0"),
            name="ck_video_track_duration_nonnegative",
        ),
        sa.CheckConstraint(
            sa.text("detection_count >= 0"),
            name="ck_video_track_detection_count_nonnegative",
        ),
        sa.CheckConstraint(
            sa.text("tracklet_count >= 0"),
            name="ck_video_track_tracklet_count_nonnegative",
        ),
        sa.CheckConstraint(
            sa.text("match_confidence >= 0 AND match_confidence <= 1"),
            name="ck_video_track_confidence_range",
        ),
        sa.CheckConstraint(
            sa.text(
                "(status_at_processing = 'known' AND name_at_processing IS NOT NULL AND btrim(name_at_processing) != '') "
                "OR (status_at_processing IN ('anonymous', 'new_anonymous') AND name_at_processing IS NULL)"
            ),
            name="ck_video_track_status_name_consistency",
        ),
    )
    op.create_index(
        "video_track_job_ordinal_idx",
        "video_track",
        ["job_id", "track_ordinal"],
        unique=True,
    )
    op.create_index("video_track_job_id_idx", "video_track", ["job_id"])
    op.create_index("video_track_face_id_idx", "video_track", ["face_id"])

    # ------------------------------------------------------------------
    # video_tracklet
    # ------------------------------------------------------------------
    op.create_table(
        "video_tracklet",
        sa.Column("tracklet_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("video_job.job_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "track_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("video_track.track_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("tracklet_ordinal", sa.Integer, nullable=False),
        sa.Column("first_frame_index", sa.BigInteger, nullable=False),
        sa.Column("last_frame_index", sa.BigInteger, nullable=False),
        sa.Column("first_pts_ns", sa.BigInteger, nullable=False),
        sa.Column("last_pts_ns", sa.BigInteger, nullable=False),
        sa.Column("observation_count", sa.Integer, nullable=False),
        sa.Column("valid_embedding_count", sa.Integer, nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("mean_quality", sa.REAL, nullable=True),
        sa.Column("max_quality", sa.REAL, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("job_id", "tracklet_ordinal", name="uq_video_tracklet_job_ordinal"),
        sa.CheckConstraint(
            sa.text(f"state IN {str(_TRACKLET_STATE_ALLOWED)}"),
            name="ck_video_tracklet_state",
        ),
        sa.CheckConstraint(
            sa.text("first_frame_index <= last_frame_index"),
            name="ck_video_tracklet_frame_order",
        ),
        sa.CheckConstraint(
            sa.text("first_pts_ns <= last_pts_ns"),
            name="ck_video_tracklet_pts_order",
        ),
        sa.CheckConstraint(
            sa.text("observation_count >= 0"),
            name="ck_video_tracklet_observation_count_nonnegative",
        ),
        sa.CheckConstraint(
            sa.text("valid_embedding_count >= 0 AND valid_embedding_count <= observation_count"),
            name="ck_video_tracklet_embedding_count_consistent",
        ),
    )
    op.create_index(
        "video_tracklet_job_ordinal_idx",
        "video_tracklet",
        ["job_id", "tracklet_ordinal"],
        unique=True,
    )
    op.create_index("video_tracklet_track_id_idx", "video_tracklet", ["track_id"])

    # ------------------------------------------------------------------
    # appearance_interval
    # ------------------------------------------------------------------
    op.create_table(
        "appearance_interval",
        sa.Column("appearance_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("video_job.job_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "track_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("video_track.track_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("interval_index", sa.Integer, nullable=False),
        sa.Column("start_frame_index", sa.BigInteger, nullable=False),
        sa.Column("end_frame_index", sa.BigInteger, nullable=False),
        sa.Column("start_pts_ns", sa.BigInteger, nullable=False),
        sa.Column("end_pts_ns", sa.BigInteger, nullable=False),
        sa.Column("detection_count", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("track_id", "interval_index", name="uq_appearance_track_interval"),
        sa.CheckConstraint(
            sa.text("start_frame_index <= end_frame_index"),
            name="ck_appearance_frame_order",
        ),
        sa.CheckConstraint(
            sa.text("start_pts_ns <= end_pts_ns"),
            name="ck_appearance_pts_order",
        ),
        sa.CheckConstraint(
            sa.text("detection_count >= 0"),
            name="ck_appearance_detection_count_nonnegative",
        ),
    )
    op.create_index(
        "appearance_track_interval_idx",
        "appearance_interval",
        ["track_id", "interval_index"],
        unique=True,
    )
    op.create_index("appearance_track_id_idx", "appearance_interval", ["track_id"])

    # ------------------------------------------------------------------
    # video_timeline_chunk
    # ------------------------------------------------------------------
    op.create_table(
        "video_timeline_chunk",
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("video_job.job_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("artifact_kind", sa.String(32), nullable=False),
        sa.Column("sequence_no", sa.Integer, nullable=False),
        sa.Column("start_pts_ns", sa.BigInteger, nullable=False),
        sa.Column("end_pts_ns", sa.BigInteger, nullable=False),
        sa.Column("bucket", sa.String(255), nullable=False),
        sa.Column("object_key", sa.String(1024), nullable=False),
        sa.Column("content_sha256", sa.CHAR(64), nullable=False),
        sa.Column("size_bytes", sa.BigInteger, nullable=False),
        sa.Column("record_count", sa.Integer, nullable=False),
        sa.Column("schema_version", sa.String(32), nullable=False),
        sa.Column("compression", sa.String(32), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("job_id", "artifact_kind", "sequence_no", name="uq_video_timeline_chunk_job_kind_seq"),
        sa.UniqueConstraint("bucket", "object_key", name="uq_video_timeline_chunk_bucket_key"),
        sa.CheckConstraint(
            sa.text(f"artifact_kind IN {str(_ARTIFACT_KIND_ALLOWED)}"),
            name="ck_video_timeline_chunk_artifact_kind",
        ),
        sa.CheckConstraint(
            sa.text("size_bytes >= 0"),
            name="ck_video_timeline_chunk_size_nonnegative",
        ),
        sa.CheckConstraint(
            sa.text("record_count >= 0"),
            name="ck_video_timeline_chunk_record_count_nonnegative",
        ),
    )
    op.create_index(
        "video_timeline_chunk_job_kind_seq_idx",
        "video_timeline_chunk",
        ["job_id", "artifact_kind", "sequence_no"],
        unique=True,
    )
    op.create_index(
        "video_timeline_chunk_bucket_key_unique_idx",
        "video_timeline_chunk",
        ["bucket", "object_key"],
        unique=True,
    )

    # ------------------------------------------------------------------
    # video_track_sample
    # ------------------------------------------------------------------
    op.create_table(
        "video_track_sample",
        sa.Column(
            "track_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("video_track.track_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "sample_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("face_sample.sample_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("sample_rank", sa.Integer, nullable=False),
        sa.Column("quality_score", sa.REAL, nullable=False),
        sa.Column("purpose", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("track_id", "sample_id"),
        sa.UniqueConstraint("track_id", "sample_rank", name="uq_video_track_sample_track_rank"),
        sa.CheckConstraint(
            sa.text(f"purpose IN {str(_SAMPLE_PURPOSE_ALLOWED)}"),
            name="ck_video_track_sample_purpose",
        ),
        sa.CheckConstraint(
            sa.text("quality_score >= 0 AND quality_score <= 1"),
            name="ck_video_track_sample_quality_range",
        ),
        sa.CheckConstraint(
            sa.text("sample_rank >= 0"),
            name="ck_video_track_sample_rank_nonnegative",
        ),
    )
    op.create_index(
        "video_track_sample_track_rank_idx",
        "video_track_sample",
        ["track_id", "sample_rank"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("video_track_sample_track_rank_idx", table_name="video_track_sample")
    op.drop_table("video_track_sample")

    op.drop_index("video_timeline_chunk_bucket_key_unique_idx", table_name="video_timeline_chunk")
    op.drop_index("video_timeline_chunk_job_kind_seq_idx", table_name="video_timeline_chunk")
    op.drop_table("video_timeline_chunk")

    op.drop_index("appearance_track_id_idx", table_name="appearance_interval")
    op.drop_index("appearance_track_interval_idx", table_name="appearance_interval")
    op.drop_table("appearance_interval")

    op.drop_index("video_tracklet_track_id_idx", table_name="video_tracklet")
    op.drop_index("video_tracklet_job_ordinal_idx", table_name="video_tracklet")
    op.drop_table("video_tracklet")

    op.drop_index("video_track_face_id_idx", table_name="video_track")
    op.drop_index("video_track_job_id_idx", table_name="video_track")
    op.drop_index("video_track_job_ordinal_idx", table_name="video_track")
    op.drop_table("video_track")

    op.drop_column("video_job", "result_schema_version")
    op.drop_column("video_job", "result_manifest_sha256")
    op.drop_column("video_job", "result_manifest_key")
    op.drop_column("video_job", "result_manifest_bucket")

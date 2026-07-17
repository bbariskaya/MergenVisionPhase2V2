"""Video control-plane schema.

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-17 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_PROCESS_STATUS_ALLOWED = (
    "processing",
    "completed",
    "failed",
    "cancelled",
)
_PROCESS_TYPE_ALLOWED = (
    "image_recognize",
    "face_enroll",
    "face_delete",
    "video_recognize",
)
_VIDEO_ASSET_STATE_ALLOWED = (
    "uploading",
    "validating",
    "ready",
    "rejected",
    "deleting",
    "deleted",
)
_VIDEO_JOB_STATE_ALLOWED = (
    "pending",
    "processing",
    "cancelling",
    "completed",
    "failed",
    "cancelled",
)
_VIDEO_JOB_STAGE_ALLOWED = (
    "queued",
    "download",
    "probe",
    "decode_infer",
    "track_reconcile",
    "persist",
    "finalize",
    "cleanup",
)
_SAMPLING_MODE_ALLOWED = (
    "every_frame",
    "every_n_frames",
    "frames_per_second",
)
_IDEMPOTENCY_STATE_ALLOWED = (
    "in_progress",
    "completed",
    "failed",
)
_EVENT_SEVERITY_ALLOWED = (
    "info",
    "warning",
    "error",
)
_OUTBOX_STATE_ALLOWED = (
    "pending",
    "processing",
    "succeeded",
    "failed",
    "dead_letter",
)


def _ts_now() -> sa.schema.DefaultClause | sa.TextClause:
    return sa.text("NOW()")


def _empty_jsonb() -> sa.schema.DefaultClause | sa.TextClause:
    return sa.text("'{}'::jsonb")


def upgrade() -> None:
    # ------------------------------------------------------------------
    # Extend process_record for Phase 2
    # ------------------------------------------------------------------
    op.add_column(
        "process_record",
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
    )

    for constraint_name in ("ck_process_record_status", "ck_process_record_type", "ck_process_record_lifecycle"):
        op.drop_constraint(constraint_name, "process_record", type_="check")

    op.create_check_constraint(
        "ck_process_record_status",
        "process_record",
        sa.text(
            f"status IN {str(_PROCESS_STATUS_ALLOWED).replace(chr(39), chr(39))}"
        ),
    )
    op.create_check_constraint(
        "ck_process_record_type",
        "process_record",
        sa.text(
            f"process_type IN {str(_PROCESS_TYPE_ALLOWED).replace(chr(39), chr(39))}"
        ),
    )
    op.create_check_constraint(
        "ck_process_record_lifecycle",
        "process_record",
        sa.text(
            "(status = 'processing' AND completed_at IS NULL AND face_count IS NULL AND error_code IS NULL) "
            "OR (status = 'completed' AND completed_at IS NOT NULL AND face_count IS NOT NULL AND face_count >= 0 AND error_code IS NULL) "
            "OR (status = 'failed' AND completed_at IS NOT NULL AND error_code IS NOT NULL AND btrim(error_code) != '') "
            "OR (status = 'cancelled' AND cancelled_at IS NOT NULL)"
        ),
    )

    # ------------------------------------------------------------------
    # video_asset
    # ------------------------------------------------------------------
    op.create_table(
        "video_asset",
        sa.Column("video_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("upload_session_id", postgresql.UUID(as_uuid=True), nullable=False, unique=True),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("staging_bucket", sa.String(255), nullable=True),
        sa.Column("staging_object_key", sa.String(1024), nullable=True),
        sa.Column("bucket", sa.String(255), nullable=True),
        sa.Column("object_key", sa.String(1024), nullable=True),
        sa.Column("content_sha256", sa.CHAR(64), nullable=True),
        sa.Column("size_bytes", sa.BigInteger, nullable=True),
        sa.Column("content_type", sa.String(255), nullable=True),
        sa.Column("container_format", sa.String(64), nullable=True),
        sa.Column("video_codec", sa.String(64), nullable=True),
        sa.Column("pixel_format", sa.String(64), nullable=True),
        sa.Column("display_width", sa.Integer, nullable=True),
        sa.Column("display_height", sa.Integer, nullable=True),
        sa.Column("rotation_degrees", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("duration_ns", sa.BigInteger, nullable=True),
        sa.Column("time_base_num", sa.Integer, nullable=True),
        sa.Column("time_base_den", sa.Integer, nullable=True),
        sa.Column("nominal_fps_num", sa.Integer, nullable=True),
        sa.Column("nominal_fps_den", sa.Integer, nullable=True),
        sa.Column("total_frames", sa.BigInteger, nullable=True),
        sa.Column("retention_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failure_code", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("ready_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer, nullable=False, server_default=sa.text("1")),
        sa.CheckConstraint(
            sa.text(
                f"state IN {str(_VIDEO_ASSET_STATE_ALLOWED)}"
            ),
            name="ck_video_asset_state",
        ),
        sa.CheckConstraint(sa.text("size_bytes IS NULL OR size_bytes >= 0"), name="ck_video_asset_size_nonnegative"),
        sa.CheckConstraint(sa.text("display_width IS NULL OR display_width > 0"), name="ck_video_asset_width_positive"),
        sa.CheckConstraint(sa.text("display_height IS NULL OR display_height > 0"), name="ck_video_asset_height_positive"),
        sa.CheckConstraint(sa.text("duration_ns IS NULL OR duration_ns >= 0"), name="ck_video_asset_duration_nonnegative"),
        sa.CheckConstraint(sa.text("time_base_den IS NULL OR time_base_den > 0"), name="ck_video_asset_time_base_den_positive"),
        sa.CheckConstraint(sa.text("nominal_fps_den IS NULL OR nominal_fps_den > 0"), name="ck_video_asset_fps_den_positive"),
        sa.CheckConstraint(
            sa.text(
                "state != 'ready' OR ("
                "bucket IS NOT NULL AND btrim(bucket) != '' "
                "AND object_key IS NOT NULL AND btrim(object_key) != '' "
                "AND content_sha256 IS NOT NULL AND btrim(content_sha256) != '' "
                "AND size_bytes IS NOT NULL "
                "AND container_format IS NOT NULL AND btrim(container_format) != '' "
                "AND video_codec IS NOT NULL AND btrim(video_codec) != '' "
                "AND display_width IS NOT NULL AND display_width > 0 "
                "AND display_height IS NOT NULL AND display_height > 0 "
                "AND duration_ns IS NOT NULL AND duration_ns >= 0 "
                "AND time_base_num IS NOT NULL AND time_base_den IS NOT NULL "
                "AND retention_until IS NOT NULL)"
            ),
            name="ck_video_asset_ready_requires_fields",
        ),
    )
    op.create_index(
        "video_asset_bucket_key_partial_idx",
        "video_asset",
        ["bucket", "object_key"],
        unique=True,
        postgresql_where=sa.text("bucket IS NOT NULL AND object_key IS NOT NULL"),
    )
    op.create_index(
        "video_asset_retention_idx",
        "video_asset",
        ["state", "retention_until"],
    )

    # ------------------------------------------------------------------
    # video_job
    # ------------------------------------------------------------------
    op.create_table(
        "video_job",
        sa.Column("job_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "video_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("video_asset.video_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "process_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("process_record.process_id", ondelete="RESTRICT"),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "retry_of_job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("video_job.job_id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("stage", sa.String(32), nullable=False),
        sa.Column("progress_percent", sa.SmallInteger, nullable=False, server_default=sa.text("0")),
        sa.Column("sampling_mode", sa.String(32), nullable=False),
        sa.Column("every_n_frames", sa.Integer, nullable=True),
        sa.Column("frames_per_second", sa.Numeric, nullable=True),
        sa.Column("processed_frames", sa.BigInteger, nullable=False, server_default=sa.text("0")),
        sa.Column("sampled_frames", sa.BigInteger, nullable=False, server_default=sa.text("0")),
        sa.Column("detected_observations", sa.BigInteger, nullable=False, server_default=sa.text("0")),
        sa.Column("person_count", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("lease_owner", sa.String(255), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempt_count", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("max_attempts", sa.Integer, nullable=False),
        sa.Column("cancellation_requested", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer, nullable=False, server_default=sa.text("1")),
        sa.CheckConstraint(
            sa.text(f"state IN {str(_VIDEO_JOB_STATE_ALLOWED)}"),
            name="ck_video_job_state",
        ),
        sa.CheckConstraint(
            sa.text(f"stage IN {str(_VIDEO_JOB_STAGE_ALLOWED)}"),
            name="ck_video_job_stage",
        ),
        sa.CheckConstraint(
            sa.text("progress_percent >= 0 AND progress_percent <= 100"),
            name="ck_video_job_progress_range",
        ),
        sa.CheckConstraint(sa.text("processed_frames >= 0"), name="ck_video_job_processed_frames_nonneg"),
        sa.CheckConstraint(sa.text("sampled_frames >= 0"), name="ck_video_job_sampled_frames_nonneg"),
        sa.CheckConstraint(sa.text("detected_observations >= 0"), name="ck_video_job_observations_nonneg"),
        sa.CheckConstraint(sa.text("person_count >= 0"), name="ck_video_job_person_count_nonneg"),
        sa.CheckConstraint(sa.text("attempt_count >= 0"), name="ck_video_job_attempt_count_nonneg"),
        sa.CheckConstraint(sa.text("max_attempts >= 1"), name="ck_video_job_max_attempts_positive"),
        sa.CheckConstraint(
            sa.text(
                "(sampling_mode = 'every_frame' AND every_n_frames IS NULL AND frames_per_second IS NULL) "
                "OR (sampling_mode = 'every_n_frames' AND every_n_frames IS NOT NULL AND every_n_frames > 0 AND frames_per_second IS NULL) "
                "OR (sampling_mode = 'frames_per_second' AND frames_per_second IS NOT NULL AND frames_per_second > 0 AND every_n_frames IS NULL)"
            ),
            name="ck_video_job_sampling_mode",
        ),
        sa.CheckConstraint(
            sa.text("state != 'completed' OR (progress_percent = 100 AND completed_at IS NOT NULL)"),
            name="ck_video_job_completed_requires_progress",
        ),
        sa.CheckConstraint(
            sa.text(
                "state != 'failed' OR (failed_at IS NOT NULL AND error_code IS NOT NULL AND btrim(error_code) != '')"
            ),
            name="ck_video_job_failed_requires_fields",
        ),
        sa.CheckConstraint(
            sa.text("state != 'cancelled' OR cancelled_at IS NOT NULL"),
            name="ck_video_job_cancelled_requires_timestamp",
        ),
        sa.CheckConstraint(
            sa.text(
                "state IN ('processing', 'cancelling') "
                "OR (lease_owner IS NULL AND lease_expires_at IS NULL AND heartbeat_at IS NULL)"
            ),
            name="ck_video_job_non_processing_no_lease",
        ),
    )
    op.create_index(
        "video_job_pending_claim_idx",
        "video_job",
        ["state", "available_at", "created_at"],
        postgresql_where=sa.text("state = 'pending'"),
    )
    op.create_index(
        "video_job_lease_recovery_idx",
        "video_job",
        ["state", "lease_expires_at"],
        postgresql_where=sa.text("state IN ('processing', 'cancelling')"),
    )
    op.create_index("video_job_video_id_idx", "video_job", ["video_id"])
    op.create_index("video_job_process_id_idx", "video_job", ["process_id"])
    op.create_index("video_job_retry_of_job_id_idx", "video_job", ["retry_of_job_id"])

    # ------------------------------------------------------------------
    # idempotency_record
    # ------------------------------------------------------------------
    op.create_table(
        "idempotency_record",
        sa.Column("scope", sa.String(64), nullable=False),
        sa.Column("key_hash", sa.CHAR(64), nullable=False),
        sa.Column("request_hash", sa.CHAR(64), nullable=True),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("resource_type", sa.String(64), nullable=True),
        sa.Column("resource_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("response_status", sa.Integer, nullable=True),
        sa.Column("response_snapshot", postgresql.JSONB, nullable=False, server_default=_empty_jsonb()),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("scope", "key_hash"),
        sa.CheckConstraint(
            sa.text(f"state IN {str(_IDEMPOTENCY_STATE_ALLOWED)}"),
            name="ck_idempotency_record_state",
        ),
    )
    op.create_index("idempotency_record_resource_idx", "idempotency_record", ["resource_type", "resource_id"])
    op.create_index("idempotency_record_expires_at_idx", "idempotency_record", ["expires_at"])

    # ------------------------------------------------------------------
    # process_event
    # ------------------------------------------------------------------
    op.create_table(
        "process_event",
        sa.Column("event_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "process_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("process_record.process_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("video_job.job_id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.Column("sequence_no", sa.Integer, nullable=False),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("payload", postgresql.JSONB, nullable=False, server_default=_empty_jsonb()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("process_id", "sequence_no", name="uq_process_event_process_sequence"),
        sa.CheckConstraint(
            sa.text(f"severity IN {str(_EVENT_SEVERITY_ALLOWED)}"),
            name="ck_process_event_severity",
        ),
    )
    op.create_index("process_event_process_id_seq_idx", "process_event", ["process_id", "sequence_no"])

    # ------------------------------------------------------------------
    # outbox_event
    # ------------------------------------------------------------------
    op.create_table(
        "outbox_event",
        sa.Column("outbox_event_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("aggregate_type", sa.String(64), nullable=False),
        sa.Column("aggregate_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("dedupe_key", sa.String(512), nullable=False, unique=True),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("attempt_count", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("max_attempts", sa.Integer, nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("locked_by", sa.String(255), nullable=True),
        sa.Column("locked_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("payload", postgresql.JSONB, nullable=False, server_default=_empty_jsonb()),
        sa.Column("last_error_code", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("succeeded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            sa.text(f"state IN {str(_OUTBOX_STATE_ALLOWED)}"),
            name="ck_outbox_event_state",
        ),
        sa.CheckConstraint(sa.text("attempt_count >= 0"), name="ck_outbox_event_attempt_nonneg"),
        sa.CheckConstraint(sa.text("max_attempts >= 1"), name="ck_outbox_event_max_attempts_positive"),
    )
    op.create_index(
        "outbox_event_claim_idx",
        "outbox_event",
        ["state", "available_at", "created_at"],
        postgresql_where=sa.text("state = 'pending'"),
    )
    op.create_index("outbox_event_aggregate_idx", "outbox_event", ["aggregate_type", "aggregate_id"])


def downgrade() -> None:
    # Forward-only for Phase 2 additions.
    op.drop_index("outbox_event_claim_idx", table_name="outbox_event")
    op.drop_index("outbox_event_aggregate_idx", table_name="outbox_event")
    op.drop_table("outbox_event")

    op.drop_index("process_event_process_id_seq_idx", table_name="process_event")
    op.drop_table("process_event")

    op.drop_index("idempotency_record_resource_idx", table_name="idempotency_record")
    op.drop_index("idempotency_record_expires_at_idx", table_name="idempotency_record")
    op.drop_table("idempotency_record")

    op.drop_index("video_job_retry_of_job_id_idx", table_name="video_job")
    op.drop_index("video_job_process_id_idx", table_name="video_job")
    op.drop_index("video_job_video_id_idx", table_name="video_job")
    op.drop_index("video_job_lease_recovery_idx", table_name="video_job")
    op.drop_index("video_job_pending_claim_idx", table_name="video_job")
    op.drop_table("video_job")

    op.drop_index("video_asset_retention_idx", table_name="video_asset")
    op.drop_index("video_asset_bucket_key_partial_idx", table_name="video_asset")
    op.drop_table("video_asset")

    for constraint_name in ("ck_process_record_lifecycle", "ck_process_record_type", "ck_process_record_status"):
        op.drop_constraint(constraint_name, "process_record", type_="check")
    op.drop_column("process_record", "cancelled_at")

    # Restore migration 0002 constraints
    op.create_check_constraint(
        "ck_process_record_status",
        "process_record",
        sa.text("status IN ('processing', 'completed', 'failed')"),
    )
    op.create_check_constraint(
        "ck_process_record_type",
        "process_record",
        sa.text("process_type IN ('image_recognize', 'face_enroll', 'face_delete')"),
    )
    op.create_check_constraint(
        "ck_process_record_lifecycle",
        "process_record",
        sa.text(
            "(status = 'processing' AND completed_at IS NULL AND face_count IS NULL AND error_code IS NULL) "
            "OR (status = 'completed' AND completed_at IS NOT NULL AND face_count IS NOT NULL AND face_count >= 0 AND error_code IS NULL) "
            "OR (status = 'failed' AND completed_at IS NOT NULL AND error_code IS NOT NULL AND btrim(error_code) != '')"
        ),
    )

"""Phase 2 contract hardening: terminal states, leases, cross-job FKs, evidence.

Revision ID: 0005
Revises: 0004
Create Date: 2026-07-17 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # process_record: replace lifecycle with precise terminal contracts
    # ------------------------------------------------------------------
    op.add_column(
        "process_record",
        sa.Column("failed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute("ALTER TABLE process_record DROP CONSTRAINT IF EXISTS ck_process_record_lifecycle")

    # Repair any rows created under the previous lifecycle contract.
    op.execute("""
        UPDATE process_record
        SET failed_at = completed_at,
            completed_at = NULL,
            cancelled_at = NULL,
            face_count = NULL
        WHERE status = 'failed' AND failed_at IS NULL
    """)
    op.execute("""
        UPDATE process_record
        SET completed_at = NULL,
            failed_at = NULL,
            error_code = NULL,
            face_count = NULL
        WHERE status = 'cancelled'
    """)
    op.execute("""
        UPDATE process_record
        SET failed_at = NULL,
            cancelled_at = NULL,
            error_code = NULL
        WHERE status = 'completed'
    """)

    op.create_check_constraint(
        "ck_process_record_terminal_completed",
        "process_record",
        sa.text(
            "status != 'completed' OR ("
            "completed_at IS NOT NULL "
            "AND failed_at IS NULL "
            "AND cancelled_at IS NULL "
            "AND error_code IS NULL "
            "AND face_count IS NOT NULL AND face_count >= 0)"
        ),
    )
    op.create_check_constraint(
        "ck_process_record_terminal_failed",
        "process_record",
        sa.text(
            "status != 'failed' OR ("
            "failed_at IS NOT NULL "
            "AND error_code IS NOT NULL AND btrim(error_code) != '' "
            "AND completed_at IS NULL "
            "AND cancelled_at IS NULL "
            "AND face_count IS NULL)"
        ),
    )
    op.create_check_constraint(
        "ck_process_record_terminal_cancelled",
        "process_record",
        sa.text(
            "status != 'cancelled' OR ("
            "cancelled_at IS NOT NULL "
            "AND completed_at IS NULL "
            "AND failed_at IS NULL "
            "AND error_code IS NULL "
            "AND face_count IS NULL)"
        ),
    )
    op.create_check_constraint(
        "ck_process_record_terminal_timestamps_disjoint",
        "process_record",
        sa.text(
            "(completed_at IS NULL OR failed_at IS NULL) "
            "AND (completed_at IS NULL OR cancelled_at IS NULL) "
            "AND (failed_at IS NULL OR cancelled_at IS NULL)"
        ),
    )

    # ------------------------------------------------------------------
    # video_job: lease token and attempt/manifest contracts
    # ------------------------------------------------------------------
    op.add_column(
        "video_job",
        sa.Column("lease_token", postgresql.UUID(as_uuid=True), nullable=True),
    )
    # Replace the 0003 lease check with the token-aware version.
    op.execute("ALTER TABLE video_job DROP CONSTRAINT IF EXISTS ck_video_job_non_processing_no_lease")

    op.create_check_constraint(
        "ck_video_job_attempt_bounds",
        "video_job",
        sa.text("attempt_count >= 0 AND max_attempts > 0 AND attempt_count <= max_attempts"),
    )
    op.create_check_constraint(
        "ck_video_job_processing_requires_lease",
        "video_job",
        sa.text(
            "state NOT IN ('processing', 'cancelling') OR ("
            "lease_owner IS NOT NULL AND lease_token IS NOT NULL "
            "AND lease_expires_at IS NOT NULL AND heartbeat_at IS NOT NULL)"
        ),
    )
    op.create_check_constraint(
        "ck_video_job_non_processing_no_lease",
        "video_job",
        sa.text(
            "state IN ('processing', 'cancelling') OR ("
            "lease_owner IS NULL AND lease_token IS NULL "
            "AND lease_expires_at IS NULL AND heartbeat_at IS NULL)"
        ),
    )
    op.create_check_constraint(
        "ck_video_job_completed_requires_manifest",
        "video_job",
        sa.text(
            "state != 'completed' OR ("
            "progress_percent = 100 "
            "AND completed_at IS NOT NULL "
            "AND result_manifest_bucket IS NOT NULL AND btrim(result_manifest_bucket) != '' "
            "AND result_manifest_key IS NOT NULL AND btrim(result_manifest_key) != '' "
            "AND result_manifest_sha256 IS NOT NULL AND btrim(result_manifest_sha256) != '' "
            "AND result_schema_version IS NOT NULL AND btrim(result_schema_version) != '')"
        ),
    )

    # ------------------------------------------------------------------
    # video_track: enforce job/track ownership and match evidence ranges
    # ------------------------------------------------------------------
    op.create_unique_constraint("uq_video_track_job_track", "video_track", ["job_id", "track_id"])

    op.create_check_constraint(
        "ck_video_track_ordinal_nonnegative",
        "video_track",
        sa.text("track_ordinal >= 0"),
    )
    op.create_check_constraint(
        "ck_video_track_top1_range",
        "video_track",
        sa.text("top1_score IS NULL OR (top1_score >= -1 AND top1_score <= 1)"),
    )
    op.create_check_constraint(
        "ck_video_track_top2_range",
        "video_track",
        sa.text("top2_score IS NULL OR (top2_score >= -1 AND top2_score <= 1)"),
    )
    op.create_check_constraint(
        "ck_video_track_threshold_range",
        "video_track",
        sa.text("threshold_used IS NULL OR (threshold_used >= -1 AND threshold_used <= 1)"),
    )
    op.create_check_constraint(
        "ck_video_track_margin_nonnegative",
        "video_track",
        sa.text("margin_score IS NULL OR margin_score >= 0"),
    )
    op.create_check_constraint(
        "ck_video_track_top_order",
        "video_track",
        sa.text("top1_score IS NULL OR top2_score IS NULL OR top1_score >= top2_score"),
    )

    # ------------------------------------------------------------------
    # video_tracklet: ordinal + cross-job FK to video_track(job_id, track_id)
    # ------------------------------------------------------------------
    op.create_check_constraint(
        "ck_video_tracklet_ordinal_nonnegative",
        "video_tracklet",
        sa.text("tracklet_ordinal >= 0"),
    )
    op.create_foreign_key(
        "fk_video_tracklet_job_track",
        "video_tracklet",
        "video_track",
        ["job_id", "track_id"],
        ["job_id", "track_id"],
        ondelete="RESTRICT",
    )

    # ------------------------------------------------------------------
    # appearance_interval: cross-job FK to video_track(job_id, track_id)
    # ------------------------------------------------------------------
    op.create_foreign_key(
        "fk_appearance_interval_job_track",
        "appearance_interval",
        "video_track",
        ["job_id", "track_id"],
        ["job_id", "track_id"],
        ondelete="RESTRICT",
    )

    # ------------------------------------------------------------------
    # video_timeline_chunk: sequence and PTS contracts
    # ------------------------------------------------------------------
    op.create_check_constraint(
        "ck_video_timeline_chunk_sequence_nonnegative",
        "video_timeline_chunk",
        sa.text("sequence_no >= 0"),
    )
    op.create_check_constraint(
        "ck_video_timeline_chunk_pts_order",
        "video_timeline_chunk",
        sa.text("start_pts_ns <= end_pts_ns"),
    )

    # ------------------------------------------------------------------
    # process_event: append-only audit stream for process/job lifetime
    # ------------------------------------------------------------------
    op.create_table(
        "process_event",
        sa.Column("event_id", postgresql.UUID(as_uuid=True), nullable=False),
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
        sa.Column(
            "payload",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("event_id"),
        sa.UniqueConstraint("process_id", "sequence_no", name="uq_process_event_process_sequence"),
        sa.CheckConstraint(
            "severity IN ('info', 'warning', 'error')",
            name="ck_process_event_severity",
        ),
    )
    op.create_index(
        "process_event_process_id_idx",
        "process_event",
        ["process_id"],
    )
    op.create_index(
        "process_event_job_id_idx",
        "process_event",
        ["job_id"],
    )

    # ------------------------------------------------------------------
    # outbox_event: durable cross-store/out-process notifications
    # ------------------------------------------------------------------
    op.create_table(
        "outbox_event",
        sa.Column("outbox_event_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("aggregate_type", sa.String(64), nullable=False),
        sa.Column("aggregate_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("dedupe_key", sa.String(512), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column(
            "attempt_count",
            sa.Integer,
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("max_attempts", sa.Integer, nullable=False),
        sa.Column(
            "available_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column("locked_by", sa.String(255), nullable=True),
        sa.Column("locked_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "payload",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("last_error_code", sa.String(64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("succeeded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("outbox_event_id"),
        sa.UniqueConstraint("dedupe_key", name="uq_outbox_event_dedupe_key"),
        sa.CheckConstraint(
            "state IN ('pending', 'processing', 'succeeded', 'failed', 'dead_letter')",
            name="ck_outbox_event_state",
        ),
        sa.CheckConstraint("attempt_count >= 0", name="ck_outbox_event_attempt_nonneg"),
        sa.CheckConstraint("max_attempts >= 1", name="ck_outbox_event_max_attempts_positive"),
    )
    op.create_index(
        "outbox_event_dispatch_idx",
        "outbox_event",
        ["state", "available_at"],
    )


def downgrade() -> None:
    # Forward-only for Phase 2 contract hardening.
    op.execute("DROP INDEX IF EXISTS outbox_event_dispatch_idx")
    op.execute("DROP TABLE IF EXISTS outbox_event")
    op.execute("DROP INDEX IF EXISTS process_event_job_id_idx")
    op.execute("DROP INDEX IF EXISTS process_event_process_id_idx")
    op.execute("DROP TABLE IF EXISTS process_event")

    op.drop_constraint("ck_video_timeline_chunk_pts_order", "video_timeline_chunk", type_="check")
    op.drop_constraint("ck_video_timeline_chunk_sequence_nonnegative", "video_timeline_chunk", type_="check")

    op.drop_constraint("fk_appearance_interval_job_track", "appearance_interval", type_="foreignkey")
    op.drop_constraint("fk_video_tracklet_job_track", "video_tracklet", type_="foreignkey")
    op.drop_constraint("ck_video_tracklet_ordinal_nonnegative", "video_tracklet", type_="check")

    op.drop_constraint("ck_video_track_top_order", "video_track", type_="check")
    op.drop_constraint("ck_video_track_margin_nonnegative", "video_track", type_="check")
    op.drop_constraint("ck_video_track_threshold_range", "video_track", type_="check")
    op.drop_constraint("ck_video_track_top2_range", "video_track", type_="check")
    op.drop_constraint("ck_video_track_top1_range", "video_track", type_="check")
    op.drop_constraint("ck_video_track_ordinal_nonnegative", "video_track", type_="check")
    op.drop_constraint("uq_video_track_job_track", "video_track", type_="unique")

    op.drop_constraint("ck_video_job_completed_requires_manifest", "video_job", type_="check")
    op.drop_constraint("ck_video_job_non_processing_no_lease", "video_job", type_="check")
    op.drop_constraint("ck_video_job_processing_requires_lease", "video_job", type_="check")
    op.drop_constraint("ck_video_job_attempt_bounds", "video_job", type_="check")
    op.drop_column("video_job", "lease_token")

    op.drop_constraint("ck_process_record_terminal_timestamps_disjoint", "process_record", type_="check")
    op.drop_constraint("ck_process_record_terminal_cancelled", "process_record", type_="check")
    op.drop_constraint("ck_process_record_terminal_failed", "process_record", type_="check")
    op.drop_constraint("ck_process_record_terminal_completed", "process_record", type_="check")
    op.create_check_constraint(
        "ck_process_record_lifecycle",
        "process_record",
        sa.text(
            "(status = 'processing' AND completed_at IS NULL AND failed_at IS NULL "
            "AND cancelled_at IS NULL AND face_count IS NULL AND error_code IS NULL) "
            "OR (status = 'completed' AND completed_at IS NOT NULL AND failed_at IS NULL "
            "AND cancelled_at IS NULL AND face_count IS NOT NULL AND face_count >= 0 AND error_code IS NULL) "
            "OR (status = 'failed' AND failed_at IS NOT NULL AND error_code IS NOT NULL "
            "AND btrim(error_code) != '' AND completed_at IS NULL AND cancelled_at IS NULL) "
            "OR (status = 'cancelled' AND cancelled_at IS NOT NULL AND completed_at IS NULL "
            "AND failed_at IS NULL AND error_code IS NULL)"
        ),
    )
    op.drop_column("process_record", "failed_at")

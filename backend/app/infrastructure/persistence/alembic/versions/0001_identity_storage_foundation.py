"""Identity storage foundation.

Revision ID: 0001
Revises:
Create Date: 2026-07-16 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. face_identity
    op.create_table(
        "face_identity",
        sa.Column("face_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("display_name", sa.String(255), nullable=True),
        sa.Column(
            "identity_metadata",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("version", sa.Integer, nullable=False, server_default=sa.text("1")),
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
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("status IN ('anonymous', 'known')", name="ck_face_identity_status"),
    )
    op.create_index(
        "face_identity_status_is_active_idx",
        "face_identity",
        ["status", "is_active"],
    )
    op.create_index(
        "face_identity_created_at_idx",
        "face_identity",
        ["created_at"],
    )

    # 2. process_record
    op.create_table(
        "process_record",
        sa.Column("process_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("process_type", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("face_count", sa.Integer, nullable=True),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column(
            "details",
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
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('processing', 'completed', 'failed')",
            name="ck_process_record_status",
        ),
    )
    op.create_index(
        "process_record_status_created_at_idx",
        "process_record",
        ["status", "created_at"],
    )
    op.create_index(
        "process_record_process_type_created_at_idx",
        "process_record",
        ["process_type", "created_at"],
    )

    # 3. face_sample
    op.create_table(
        "face_sample",
        sa.Column("sample_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "face_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("face_identity.face_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("bucket", sa.String(255), nullable=True),
        sa.Column("object_key", sa.String(1024), nullable=True),
        sa.Column("failure_code", sa.String(64), nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deactivated_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "state IN ('pending', 'active', 'failed', 'inactive')",
            name="ck_face_sample_state",
        ),
    )
    op.create_index(
        "face_sample_face_id_sample_state_idx",
        "face_sample",
        ["face_id", "state"],
    )
    op.create_index(
        "face_sample_bucket_key_unique_idx",
        "face_sample",
        ["bucket", "object_key"],
        unique=True,
        postgresql_where=sa.text("bucket IS NOT NULL AND object_key IS NOT NULL"),
    )

    # 4. recognition_result
    op.create_table(
        "recognition_result",
        sa.Column("result_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "process_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("process_record.process_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "face_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("face_identity.face_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "sample_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("face_sample.sample_id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("bounding_box", postgresql.JSONB, nullable=False),
        sa.Column("match_confidence", sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "result_metadata",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.CheckConstraint(
            "status IN ('known', 'anonymous', 'new_anonymous')",
            name="ck_recognition_result_status",
        ),
    )
    op.create_index(
        "recognition_result_process_id_result_index_idx",
        "recognition_result",
        ["process_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "recognition_result_process_id_result_index_idx",
        table_name="recognition_result",
    )
    op.drop_table("recognition_result")
    op.drop_index("face_sample_bucket_key_unique_idx", table_name="face_sample")
    op.drop_index("face_sample_face_id_sample_state_idx", table_name="face_sample")
    op.drop_table("face_sample")
    op.drop_index("process_record_process_type_created_at_idx", table_name="process_record")
    op.drop_index("process_record_status_created_at_idx", table_name="process_record")
    op.drop_table("process_record")
    op.drop_index("face_identity_created_at_idx", table_name="face_identity")
    op.drop_index("face_identity_status_is_active_idx", table_name="face_identity")
    op.drop_table("face_identity")

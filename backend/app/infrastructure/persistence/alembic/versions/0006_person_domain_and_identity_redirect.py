"""Person domain and identity redirect.

Revision ID: 0006
Revises: cf0441294c5f
Create Date: 2026-07-18 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006"
down_revision: str | Sequence[str] | None = "cf0441294c5f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # Extend process_record type enum for assign/merge flows
    # ------------------------------------------------------------------
    op.execute("ALTER TABLE process_record DROP CONSTRAINT IF EXISTS ck_process_record_type")
    op.create_check_constraint(
        "ck_process_record_type",
        "process_record",
        sa.text(
            "process_type IN ('image_recognize', 'face_enroll', 'face_delete', "
            "'video_recognize', 'face_assign')"
        ),
    )

    # ------------------------------------------------------------------
    # person table
    # ------------------------------------------------------------------
    op.create_table(
        "person",
        sa.Column("person_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("display_name", sa.String(255), nullable=False),
        sa.Column(
            "person_metadata",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
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
        sa.CheckConstraint("version >= 1", name="ck_person_version_positive"),
        sa.CheckConstraint(
            "(is_active = true AND deleted_at IS NULL) OR (is_active = false AND deleted_at IS NOT NULL)",
            name="ck_person_active_deleted",
        ),
    )
    op.create_index(
        "person_display_name_idx",
        "person",
        ["display_name"],
    )
    op.create_index(
        "person_is_active_idx",
        "person",
        ["is_active"],
    )

    # ------------------------------------------------------------------
    # Extend face_identity with person and redirect links
    # ------------------------------------------------------------------
    op.add_column(
        "face_identity",
        sa.Column(
            "person_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("person.person_id", ondelete="RESTRICT"),
            nullable=True,
        ),
    )
    op.add_column(
        "face_identity",
        sa.Column(
            "redirect_to_face_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("face_identity.face_id", ondelete="RESTRICT"),
            nullable=True,
        ),
    )

    op.create_index(
        "face_identity_person_id_idx",
        "face_identity",
        ["person_id"],
    )
    op.create_index(
        "face_identity_redirect_idx",
        "face_identity",
        ["redirect_to_face_id"],
    )

    # Drop the old known-name check so we can replace it with the person-aware version.
    op.execute("ALTER TABLE face_identity DROP CONSTRAINT IF EXISTS ck_face_identity_known_name")

    op.create_check_constraint(
        "ck_face_identity_known_requires_person",
        "face_identity",
        sa.text(
            "status != 'known' OR (person_id IS NOT NULL AND display_name IS NOT NULL AND btrim(display_name) != '')"
        ),
    )
    op.create_check_constraint(
        "ck_face_identity_redirect_inactive",
        "face_identity",
        sa.text(
            "(redirect_to_face_id IS NULL) OR (redirect_to_face_id IS NOT NULL AND is_active = false)"
        ),
    )

    # ------------------------------------------------------------------
    # Backfill: one person per existing known face_identity.
    # Use a temporary column to guarantee a 1:1 match regardless of name collisions.
    # ------------------------------------------------------------------
    op.add_column(
        "person",
        sa.Column("_source_face_id", postgresql.UUID(as_uuid=True), nullable=True),
    )

    op.execute(
        """
        INSERT INTO person (person_id, display_name, person_metadata, created_at, updated_at, _source_face_id)
        SELECT
            gen_random_uuid(),
            display_name,
            COALESCE(identity_metadata, '{}'::jsonb),
            created_at,
            updated_at,
            face_id
        FROM face_identity
        WHERE status = 'known' AND is_active = true;
        """
    )
    op.execute(
        """
        UPDATE face_identity AS fi
        SET person_id = p.person_id
        FROM person AS p
        WHERE p._source_face_id IS NOT NULL
          AND p._source_face_id = fi.face_id;
        """
    )

    op.drop_column("person", "_source_face_id")


def downgrade() -> None:
    op.drop_index("face_identity_redirect_idx", table_name="face_identity")
    op.drop_index("face_identity_person_id_idx", table_name="face_identity")
    op.drop_column("face_identity", "redirect_to_face_id")
    op.drop_column("face_identity", "person_id")

    op.drop_index("person_is_active_idx", table_name="person")
    op.drop_index("person_display_name_idx", table_name="person")
    op.drop_table("person")

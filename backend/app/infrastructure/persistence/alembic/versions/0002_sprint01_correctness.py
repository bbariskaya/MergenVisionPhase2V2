"""Sprint 01 correctness constraints and defaults.

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-16 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # face_identity
    # ------------------------------------------------------------------
    op.create_check_constraint(
        "ck_face_identity_version_positive",
        "face_identity",
        sa.text("version >= 1"),
    )
    op.create_check_constraint(
        "ck_face_identity_known_name",
        "face_identity",
        sa.text("status != 'known' OR (display_name IS NOT NULL AND btrim(display_name) != '')"),
    )
    op.create_check_constraint(
        "ck_face_identity_active_deleted",
        "face_identity",
        sa.text(
            "(is_active = true AND deleted_at IS NULL) "
            "OR (is_active = false AND deleted_at IS NOT NULL)"
        ),
    )

    # ------------------------------------------------------------------
    # process_record
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # face_sample
    # ------------------------------------------------------------------
    # Repair rows created with the incorrect default is_active=true.
    op.execute("UPDATE face_sample SET is_active = (state = 'active')")

    # Change default so newly inserted pending rows are inactive.
    op.alter_column(
        "face_sample",
        "is_active",
        existing_type=sa.Boolean(),
        server_default=sa.false(),
        existing_nullable=False,
    )

    op.create_check_constraint(
        "ck_face_sample_lifecycle",
        "face_sample",
        sa.text(
            "(state = 'pending' AND is_active = false AND bucket IS NULL AND object_key IS NULL "
            "AND activated_at IS NULL AND failure_code IS NULL AND deactivated_at IS NULL) "
            "OR (state = 'active' AND is_active = true AND bucket IS NOT NULL "
            "AND btrim(bucket) != '' AND object_key IS NOT NULL AND btrim(object_key) != '' "
            "AND activated_at IS NOT NULL AND failure_code IS NULL AND deactivated_at IS NULL) "
            "OR (state = 'failed' AND is_active = false AND failure_code IS NOT NULL "
            "AND btrim(failure_code) != '') "
            "OR (state = 'inactive' AND is_active = false AND bucket IS NOT NULL "
            "AND btrim(bucket) != '' AND object_key IS NOT NULL AND btrim(object_key) != '' "
            "AND activated_at IS NOT NULL AND deactivated_at IS NOT NULL)"
        ),
    )

    # ------------------------------------------------------------------
    # recognition_result
    # ------------------------------------------------------------------
    op.create_check_constraint(
        "ck_recognition_result_confidence",
        "recognition_result",
        sa.text("match_confidence >= 0 AND match_confidence <= 1"),
    )
    op.create_index(
        "recognition_result_face_id_idx",
        "recognition_result",
        ["face_id"],
    )
    op.create_index(
        "recognition_result_sample_id_idx",
        "recognition_result",
        ["sample_id"],
    )


def downgrade() -> None:
    # Forward-only: drop only the artifacts added by this migration.
    op.drop_constraint("ck_face_identity_version_positive", "face_identity", type_="check")
    op.drop_constraint("ck_face_identity_known_name", "face_identity", type_="check")
    op.drop_constraint("ck_face_identity_active_deleted", "face_identity", type_="check")

    op.drop_constraint("ck_process_record_type", "process_record", type_="check")
    op.drop_constraint("ck_process_record_lifecycle", "process_record", type_="check")

    op.drop_constraint("ck_face_sample_lifecycle", "face_sample", type_="check")
    op.alter_column(
        "face_sample",
        "is_active",
        existing_type=sa.Boolean(),
        server_default=sa.true(),
        existing_nullable=False,
    )

    op.drop_constraint("ck_recognition_result_confidence", "recognition_result", type_="check")
    op.drop_index("recognition_result_face_id_idx", table_name="recognition_result")
    op.drop_index("recognition_result_sample_id_idx", table_name="recognition_result")

"""fix video_tracklet unique constraint to include track_id.

Revision ID: cf0441294c5f
Revises: 0005
Create Date: 2026-07-17 21:41:44.181622

"""
from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = 'cf0441294c5f'
down_revision: str | None = '0005'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass

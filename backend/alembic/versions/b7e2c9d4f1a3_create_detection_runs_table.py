"""create detection_runs table (scheduled detection: one row per processed image)

Revision ID: b7e2c9d4f1a3
Revises: 9c4f7a1e3b2d
Create Date: 2026-10-19 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7e2c9d4f1a3'
down_revision: Union[str, Sequence[str], None] = '9c4f7a1e3b2d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'detection_runs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('image_time', sa.DateTime(), nullable=False),
        sa.Column('started_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('finished_at', sa.DateTime(), nullable=True),
        sa.Column('status', sa.Text(), server_default='running', nullable=False),
        sa.Column('detection_count', sa.Integer(), nullable=True),
        sa.Column('alert_count', sa.Integer(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('image_time'),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('detection_runs')

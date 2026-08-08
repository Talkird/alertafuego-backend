"""create false_positive_reports table

Revision ID: 2a336f408a15
Revises: 7f931a80ef9c
Create Date: 2026-08-06 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '2a336f408a15'
down_revision: Union[str, Sequence[str], None] = '7f931a80ef9c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

report_category = postgresql.ENUM(
    'industrial_activity', 'controlled_burn', 'gas_flare', 'sun_glint', 'sensor_noise', 'other',
    name='report_category',
)


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'false_positive_reports',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('detection_id', sa.Integer(), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('category', report_category, nullable=False),
        sa.Column('comment', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['detection_id'], ['detections.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('detection_id', 'user_id', name='uq_report_detection_user'),
    )
    op.create_index(
        op.f('ix_false_positive_reports_detection_id'), 'false_positive_reports', ['detection_id']
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_false_positive_reports_detection_id'), table_name='false_positive_reports')
    op.drop_table('false_positive_reports')

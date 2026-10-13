"""create user_settings table

Revision ID: 5b8e1d2c4a7f
Revises: cc2e2412d83b
Create Date: 2026-10-13 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5b8e1d2c4a7f'
down_revision: Union[str, Sequence[str], None] = 'cc2e2412d83b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'user_settings',
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('alert_email', sa.Text(), nullable=True),
        sa.Column('pending_email', sa.Text(), nullable=True),
        sa.Column('pending_token_hash', sa.Text(), nullable=True),
        sa.Column('pending_expires_at', sa.DateTime(), nullable=True),
        sa.Column('default_radius_km', sa.Float(), server_default='25', nullable=False),
        sa.Column('default_min_probability', sa.Float(), server_default='0.65', nullable=False),
        sa.CheckConstraint('default_radius_km > 0 AND default_radius_km <= 200', name='ck_user_settings_radius'),
        sa.CheckConstraint(
            'default_min_probability >= 0 AND default_min_probability <= 1', name='ck_user_settings_min_probability'
        ),
        sa.PrimaryKeyConstraint('user_id'),
        sa.UniqueConstraint('pending_token_hash'),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('user_settings')

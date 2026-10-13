"""create user_settings table, add telegram_sent_at to alert_notifications

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
        sa.Column('email_alerts', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('alert_email', sa.Text(), nullable=True),
        sa.Column('pending_email', sa.Text(), nullable=True),
        sa.Column('pending_token_hash', sa.Text(), nullable=True),
        sa.Column('pending_expires_at', sa.DateTime(), nullable=True),
        sa.Column('telegram_chat_id', sa.BigInteger(), nullable=True),
        sa.Column('telegram_name', sa.Text(), nullable=True),
        sa.Column('telegram_link_token_hash', sa.Text(), nullable=True),
        sa.Column('telegram_link_expires_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('user_id'),
        sa.UniqueConstraint('pending_token_hash'),
        sa.UniqueConstraint('telegram_chat_id'),
        sa.UniqueConstraint('telegram_link_token_hash'),
    )
    op.add_column('alert_notifications', sa.Column('telegram_sent_at', sa.DateTime(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('alert_notifications', 'telegram_sent_at')
    op.drop_table('user_settings')

"""alert areas can be a province or department (region_code) instead of a circle

Revision ID: 9c4f7a1e3b2d
Revises: 5b8e1d2c4a7f
Create Date: 2026-10-17 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9c4f7a1e3b2d'
down_revision: Union[str, Sequence[str], None] = '5b8e1d2c4a7f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('alert_areas', sa.Column('region_code', sa.Text(), nullable=True))
    op.alter_column('alert_areas', 'center', nullable=True)
    op.alter_column('alert_areas', 'radius_km', nullable=True)
    op.create_check_constraint(
        'ck_alert_area_shape',
        'alert_areas',
        '(region_code IS NULL AND center IS NOT NULL AND radius_km IS NOT NULL)'
        ' OR (region_code IS NOT NULL AND center IS NULL AND radius_km IS NULL)',
    )


def downgrade() -> None:
    """Downgrade schema. Deletes region alert areas (they have no circle to keep)."""
    op.drop_constraint('ck_alert_area_shape', 'alert_areas', type_='check')
    op.execute('DELETE FROM alert_areas WHERE region_code IS NOT NULL')
    op.alter_column('alert_areas', 'radius_km', nullable=False)
    op.alter_column('alert_areas', 'center', nullable=False)
    op.drop_column('alert_areas', 'region_code')

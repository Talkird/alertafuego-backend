"""alert areas can be a polygon or a province or department, not only a circle

Revision ID: 9c4f7a1e3b2d
Revises: 5b8e1d2c4a7f
Create Date: 2026-10-17 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import geoalchemy2


# revision identifiers, used by Alembic.
revision: str = '9c4f7a1e3b2d'
down_revision: Union[str, Sequence[str], None] = '5b8e1d2c4a7f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('alert_areas', sa.Column('region_code', sa.Text(), nullable=True))
    op.add_column(
        'alert_areas',
        sa.Column(
            'polygon',
            geoalchemy2.types.Geometry(geometry_type='POLYGON', srid=4326, spatial_index=False, from_text='ST_GeomFromEWKT', name='geometry'),
            nullable=True,
        ),
    )
    op.alter_column('alert_areas', 'center', nullable=True)
    op.alter_column('alert_areas', 'radius_km', nullable=True)
    op.create_check_constraint(
        'ck_alert_area_shape',
        'alert_areas',
        'num_nonnulls(center, polygon, region_code) = 1 AND (center IS NULL) = (radius_km IS NULL)',
    )


def downgrade() -> None:
    """Downgrade schema. Deletes polygon and region alert areas (they have no circle
    to keep)."""
    op.drop_constraint('ck_alert_area_shape', 'alert_areas', type_='check')
    op.execute('DELETE FROM alert_areas WHERE center IS NULL')
    op.alter_column('alert_areas', 'radius_km', nullable=False)
    op.alter_column('alert_areas', 'center', nullable=False)
    op.drop_column('alert_areas', 'polygon')
    op.drop_column('alert_areas', 'region_code')

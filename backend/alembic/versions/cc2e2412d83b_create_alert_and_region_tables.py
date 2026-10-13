"""create alert_areas, alert_notifications and regions tables

Revision ID: cc2e2412d83b
Revises: 78d030e2a232
Create Date: 2026-10-12 21:16:51.349017

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import geoalchemy2
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'cc2e2412d83b'
down_revision: Union[str, Sequence[str], None] = '78d030e2a232'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

region_kind = postgresql.ENUM('province', 'department', name='region_kind')


def upgrade() -> None:
    """Upgrade schema."""
    # As in the detections migration, the GiST indexes on `center` and `geometry` come
    # from GeoAlchemy2's spatial_index DDL event - no explicit create_index for them.
    op.create_table(
        'alert_areas',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email', sa.Text(), nullable=False),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('center', geoalchemy2.types.Geography(geometry_type='POINT', srid=4326, dimension=2, from_text='ST_GeogFromText', name='geography', nullable=False), nullable=False),
        sa.Column('radius_km', sa.Float(), nullable=False),
        sa.Column('min_probability', sa.Float(), nullable=False),
        sa.Column('active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint('min_probability >= 0 AND min_probability <= 1', name='ck_alert_area_min_probability'),
        sa.CheckConstraint('radius_km > 0 AND radius_km <= 200', name='ck_alert_area_radius'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_alert_areas_user_id'), 'alert_areas', ['user_id'])

    op.create_table(
        'alert_notifications',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('area_id', sa.Integer(), nullable=False),
        sa.Column('detection_id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('emailed_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['area_id'], ['alert_areas.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['detection_id'], ['detections.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('area_id', 'detection_id', name='uq_alert_area_detection'),
    )
    op.create_index(op.f('ix_alert_notifications_area_id'), 'alert_notifications', ['area_id'])
    op.create_index(op.f('ix_alert_notifications_detection_id'), 'alert_notifications', ['detection_id'])

    op.create_table(
        'regions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('kind', region_kind, nullable=False),
        sa.Column('code', sa.Text(), nullable=False),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('province_code', sa.Text(), nullable=True),
        sa.Column('geometry', geoalchemy2.types.Geometry(geometry_type='MULTIPOLYGON', srid=4326, dimension=2, from_text='ST_GeomFromEWKT', name='geometry', nullable=False), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('kind', 'code', name='uq_region_kind_code'),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('regions')
    region_kind.drop(op.get_bind())
    op.drop_index(op.f('ix_alert_notifications_detection_id'), table_name='alert_notifications')
    op.drop_index(op.f('ix_alert_notifications_area_id'), table_name='alert_notifications')
    op.drop_table('alert_notifications')
    op.drop_index(op.f('ix_alert_areas_user_id'), table_name='alert_areas')
    op.drop_table('alert_areas')

"""generalize false_positive_reports into detection_reports with a verdict

Revision ID: 78d030e2a232
Revises: 217c98e19208
Create Date: 2026-08-26 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '78d030e2a232'
down_revision: Union[str, Sequence[str], None] = '217c98e19208'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

report_verdict = postgresql.ENUM('false_positive', 'confirmed_fire', name='report_verdict')


def upgrade() -> None:
    """Upgrade schema."""
    op.rename_table('false_positive_reports', 'detection_reports')

    report_verdict.create(op.get_bind())
    # Every existing row predates the verdict column, so it's a false-positive
    # report by definition - backfill via the default, then drop it since new rows
    # must always specify a verdict explicitly.
    op.add_column(
        'detection_reports',
        sa.Column('verdict', report_verdict, nullable=False, server_default='false_positive'),
    )
    op.alter_column('detection_reports', 'verdict', server_default=None)

    op.alter_column('detection_reports', 'category', nullable=True)
    op.create_check_constraint(
        'ck_category_matches_verdict',
        'detection_reports',
        "(verdict = 'false_positive' AND category IS NOT NULL) OR "
        "(verdict = 'confirmed_fire' AND category IS NULL)",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint('ck_category_matches_verdict', 'detection_reports', type_='check')
    op.alter_column('detection_reports', 'category', nullable=False)
    op.drop_column('detection_reports', 'verdict')
    report_verdict.drop(op.get_bind())

    op.rename_table('detection_reports', 'false_positive_reports')

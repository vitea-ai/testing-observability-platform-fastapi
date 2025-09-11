"""Add task_id field to evaluations table

Revision ID: add_task_id_001
Revises: fa29f95bb5da
Create Date: 2025-09-09 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'add_task_id_001'
down_revision: Union[str, None] = 'fa29f95bb5da'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add task_id column to evaluations table for Celery task tracking."""
    op.add_column('evaluations', 
        sa.Column('task_id', sa.String(length=255), nullable=True)
    )
    op.create_index(
        op.f('ix_evaluations_task_id'), 
        'evaluations', 
        ['task_id'], 
        unique=False
    )


def downgrade() -> None:
    """Remove task_id column from evaluations table."""
    op.drop_index(op.f('ix_evaluations_task_id'), table_name='evaluations')
    op.drop_column('evaluations', 'task_id')
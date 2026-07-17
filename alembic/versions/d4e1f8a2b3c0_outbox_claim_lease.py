"""outbox claim lease for multi-dispatch replicas

Revision ID: d4e1f8a2b3c0
Revises: c3d9e2f5a7b1
Create Date: 2026-07-17 18:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d4e1f8a2b3c0"
down_revision: Union[str, Sequence[str], None] = "c3d9e2f5a7b1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("pending_eval_jobs") as batch:
        batch.add_column(sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index(
        op.f("ix_pending_eval_jobs_claimed_at"),
        "pending_eval_jobs",
        ["claimed_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_pending_eval_jobs_claimed_at"), table_name="pending_eval_jobs")
    with op.batch_alter_table("pending_eval_jobs") as batch:
        batch.drop_column("claimed_at")

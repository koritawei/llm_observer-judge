"""multi_tenant_projects

Revision ID: c3d9e2f5a7b1
Revises: b2c8d1e4f6a0
Create Date: 2026-07-17 16:50:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c3d9e2f5a7b1"
down_revision: Union[str, Sequence[str], None] = "b2c8d1e4f6a0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("project_id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("tenant_id", "project_id"),
        sa.UniqueConstraint("tenant_id", "name", name="uq_project_tenant_name"),
    )

    # Add project_id to existing tables (default = 'default')
    for table in (
        "pending_eval_jobs",
        "human_annotations",
        "bad_case_attributions",
        "benchmark_reports",
        "spans",
        "eval_results",
    ):
        op.add_column(
            table,
            sa.Column(
                "project_id",
                sa.String(length=64),
                nullable=False,
                server_default="default",
            ),
        )
        op.create_index(f"ix_{table}_project_id", table, ["project_id"], unique=False)

    op.add_column(
        "prompt_meta",
        sa.Column("project_id", sa.String(length=64), nullable=True),
    )
    op.create_index(
        op.f("ix_prompt_meta_project_id"), "prompt_meta", ["project_id"], unique=False
    )

    # Rebuild attribution_checkpoints with composite key support
    op.drop_table("attribution_checkpoints")
    op.create_table(
        "attribution_checkpoints",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("project_id", sa.String(length=64), nullable=False),
        sa.Column("trace_id", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "tenant_id",
            "project_id",
            "trace_id",
            name="uq_checkpoint_tenant_project_trace",
        ),
    )
    op.create_index(
        op.f("ix_attribution_checkpoints_tenant_id"),
        "attribution_checkpoints",
        ["tenant_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_attribution_checkpoints_project_id"),
        "attribution_checkpoints",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_attribution_checkpoints_trace_id"),
        "attribution_checkpoints",
        ["trace_id"],
        unique=False,
    )

    # Unique constraints: drop old, add tenant+project scoped
    with op.batch_alter_table("pending_eval_jobs") as batch:
        batch.drop_constraint("uq_pending_tenant_trace", type_="unique")
        batch.create_unique_constraint(
            "uq_pending_tenant_project_trace",
            ["tenant_id", "project_id", "trace_id"],
        )

    with op.batch_alter_table("spans") as batch:
        batch.drop_constraint("uq_span_tenant_span", type_="unique")
        batch.create_unique_constraint(
            "uq_span_tenant_project_span",
            ["tenant_id", "project_id", "span_id"],
        )

    with op.batch_alter_table("eval_results") as batch:
        batch.drop_constraint("uq_eval_tenant_trace", type_="unique")
        batch.create_unique_constraint(
            "uq_eval_tenant_project_trace",
            ["tenant_id", "project_id", "trace_id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("eval_results") as batch:
        batch.drop_constraint("uq_eval_tenant_project_trace", type_="unique")
        batch.create_unique_constraint(
            "uq_eval_tenant_trace", ["tenant_id", "trace_id"]
        )
    with op.batch_alter_table("spans") as batch:
        batch.drop_constraint("uq_span_tenant_project_span", type_="unique")
        batch.create_unique_constraint("uq_span_tenant_span", ["tenant_id", "span_id"])
    with op.batch_alter_table("pending_eval_jobs") as batch:
        batch.drop_constraint("uq_pending_tenant_project_trace", type_="unique")
        batch.create_unique_constraint(
            "uq_pending_tenant_trace", ["tenant_id", "trace_id"]
        )

    op.drop_table("attribution_checkpoints")
    op.create_table(
        "attribution_checkpoints",
        sa.Column("trace_id", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("trace_id"),
    )

    op.drop_index(op.f("ix_prompt_meta_project_id"), table_name="prompt_meta")
    op.drop_column("prompt_meta", "project_id")

    for table in (
        "eval_results",
        "spans",
        "benchmark_reports",
        "bad_case_attributions",
        "human_annotations",
        "pending_eval_jobs",
    ):
        op.drop_index(f"ix_{table}_project_id", table_name=table)
        op.drop_column(table, "project_id")

    op.drop_table("projects")

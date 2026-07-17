"""distributed stores: spans, eval_results, prompts meta/experiments, checkpoints

Revision ID: b2c8d1e4f6a0
Revises: e79449645152
Create Date: 2026-07-17 16:40:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b2c8d1e4f6a0"
down_revision: Union[str, Sequence[str], None] = "e79449645152"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "spans",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("trace_id", sa.String(length=64), nullable=False),
        sa.Column("span_id", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tenant_id", "span_id", name="uq_span_tenant_span"),
    )
    op.create_index(op.f("ix_spans_tenant_id"), "spans", ["tenant_id"], unique=False)
    op.create_index(op.f("ix_spans_trace_id"), "spans", ["trace_id"], unique=False)
    op.create_index(op.f("ix_spans_span_id"), "spans", ["span_id"], unique=False)

    op.create_table(
        "eval_results",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("trace_id", sa.String(length=64), nullable=False),
        sa.Column("idempotency_key", sa.String(length=256), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tenant_id", "trace_id", name="uq_eval_tenant_trace"),
        sa.UniqueConstraint("idempotency_key"),
    )
    op.create_index(op.f("ix_eval_results_tenant_id"), "eval_results", ["tenant_id"], unique=False)
    op.create_index(op.f("ix_eval_results_trace_id"), "eval_results", ["trace_id"], unique=False)

    op.create_table(
        "prompt_meta",
        sa.Column("prompt_id", sa.String(length=128), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=128), nullable=True),
        sa.PrimaryKeyConstraint("prompt_id"),
    )
    op.create_index(op.f("ix_prompt_meta_tenant_id"), "prompt_meta", ["tenant_id"], unique=False)

    op.create_table(
        "prompt_experiments",
        sa.Column("experiment_id", sa.String(length=64), nullable=False),
        sa.Column("prompt_id", sa.String(length=128), nullable=False),
        sa.Column("control_version_id", sa.String(length=64), nullable=False),
        sa.Column("treatment_version_id", sa.String(length=64), nullable=False),
        sa.Column("traffic_pct", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("metric_primary", sa.String(length=64), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("conclusion", sa.Text(), nullable=True),
        sa.Column("winner_version_id", sa.String(length=64), nullable=True),
        sa.PrimaryKeyConstraint("experiment_id"),
    )
    op.create_index(
        op.f("ix_prompt_experiments_prompt_id"), "prompt_experiments", ["prompt_id"], unique=False
    )
    op.create_index(
        op.f("ix_prompt_experiments_status"), "prompt_experiments", ["status"], unique=False
    )

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

    op.create_table(
        "idempotency_keys",
        sa.Column("key", sa.String(length=256), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("key"),
    )


def downgrade() -> None:
    op.drop_table("idempotency_keys")
    op.drop_table("attribution_checkpoints")
    op.drop_index(op.f("ix_prompt_experiments_status"), table_name="prompt_experiments")
    op.drop_index(op.f("ix_prompt_experiments_prompt_id"), table_name="prompt_experiments")
    op.drop_table("prompt_experiments")
    op.drop_index(op.f("ix_prompt_meta_tenant_id"), table_name="prompt_meta")
    op.drop_table("prompt_meta")
    op.drop_index(op.f("ix_eval_results_trace_id"), table_name="eval_results")
    op.drop_index(op.f("ix_eval_results_tenant_id"), table_name="eval_results")
    op.drop_table("eval_results")
    op.drop_index(op.f("ix_spans_span_id"), table_name="spans")
    op.drop_index(op.f("ix_spans_trace_id"), table_name="spans")
    op.drop_index(op.f("ix_spans_tenant_id"), table_name="spans")
    op.drop_table("spans")

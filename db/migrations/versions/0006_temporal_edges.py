"""Add temporal validity and confidence to canonical_relations

Revision ID: 0006
Revises: 0005
Create Date: 2026-06-01
"""
from alembic import op
import sqlalchemy as sa

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "canonical_relations",
        sa.Column(
            "valid_from",
            sa.DateTime(timezone=True),
            nullable=True,  # nullable during migration; backfilled below
        ),
    )
    op.add_column(
        "canonical_relations",
        sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "canonical_relations",
        sa.Column("confidence", sa.Float(), nullable=True),
    )

    # Backfill existing rows: all current edges are valid from their creation date
    op.execute(
        "UPDATE canonical_relations SET valid_from = created_at, confidence = 1.0 "
        "WHERE valid_from IS NULL"
    )

    # Now enforce NOT NULL on valid_from and confidence
    op.alter_column("canonical_relations", "valid_from", nullable=False)
    op.alter_column("canonical_relations", "confidence", nullable=False)

    op.create_index("ix_canonical_relations_valid_from", "canonical_relations", ["valid_from"])

    # Drop obsolete tables from the old risk/simulation world
    op.drop_table("simulation_runs")
    op.drop_table("forecast_records")
    op.drop_table("world_model_states")


def downgrade() -> None:
    op.drop_index("ix_canonical_relations_valid_from", table_name="canonical_relations")
    op.drop_column("canonical_relations", "valid_from")
    op.drop_column("canonical_relations", "valid_to")
    op.drop_column("canonical_relations", "confidence")

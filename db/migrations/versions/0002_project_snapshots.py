"""project snapshots

Revision ID: 0002_project_snapshots
Revises: 0001_initial
Create Date: 2026-05-30
"""

from alembic import op
import sqlalchemy as sa


revision = "0002_project_snapshots"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "project_snapshots",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("snapshot_date", sa.Date(), nullable=False),
        sa.Column("total_documents", sa.Integer(), nullable=False),
        sa.Column("total_chunks", sa.Integer(), nullable=False),
        sa.Column("owners_detected", sa.Integer(), nullable=False),
        sa.Column("actions_detected", sa.Integer(), nullable=False),
        sa.Column("dates_detected", sa.Integer(), nullable=False),
        sa.Column("risks_detected", sa.Integer(), nullable=False),
        sa.Column("blockers_detected", sa.Integer(), nullable=False),
        sa.Column("dependencies_detected", sa.Integer(), nullable=False),
        sa.Column("owner_coverage_ratio", sa.Float(), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("risk_level", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_project_snapshots_run_id", "project_snapshots", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_project_snapshots_run_id", "project_snapshots")
    op.drop_table("project_snapshots")

"""integrations and telemetry

Revision ID: 0003_integrations_and_telemetry
Revises: 0002_project_snapshots
Create Date: 2026-05-30
"""

from alembic import op
import sqlalchemy as sa


revision = "0003_integrations_and_telemetry"
down_revision = "0002_project_snapshots"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "integration_connections",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("base_url", sa.String(length=1024), nullable=False),
        sa.Column("auth_token", sa.Text(), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_integration_connections_provider", "integration_connections", ["provider"])

    op.create_table(
        "telemetry_snapshots",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("connection_id", sa.String(length=32), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("project_ref", sa.String(length=255), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("risk_level", sa.String(length=16), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_telemetry_snapshots_connection_id", "telemetry_snapshots", ["connection_id"])
    op.create_index("ix_telemetry_snapshots_project_ref", "telemetry_snapshots", ["project_ref"])


def downgrade() -> None:
    op.drop_index("ix_telemetry_snapshots_project_ref", "telemetry_snapshots")
    op.drop_index("ix_telemetry_snapshots_connection_id", "telemetry_snapshots")
    op.drop_table("telemetry_snapshots")
    op.drop_index("ix_integration_connections_provider", "integration_connections")
    op.drop_table("integration_connections")

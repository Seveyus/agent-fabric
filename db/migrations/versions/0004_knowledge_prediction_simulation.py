"""knowledge prediction simulation

Revision ID: 0004_knowledge_layers
Revises: 0003_integrations_and_telemetry
Create Date: 2026-05-30
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "0004_knowledge_layers"
down_revision = "0003_integrations_and_telemetry"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    tables = set(inspector.get_table_names())

    if "knowledge_nodes" not in tables:
        op.create_table(
            "knowledge_nodes",
            sa.Column("id", sa.String(length=32), primary_key=True),
            sa.Column("node_type", sa.String(length=64), nullable=False),
            sa.Column("external_ref", sa.String(length=255), nullable=False),
            sa.Column("title", sa.String(length=255), nullable=False),
            sa.Column("attributes", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
    _ensure_index(inspector, "knowledge_nodes", "ix_knowledge_nodes_node_type", ["node_type"])
    _ensure_index(inspector, "knowledge_nodes", "ix_knowledge_nodes_external_ref", ["external_ref"])

    if "knowledge_edges" not in tables:
        op.create_table(
            "knowledge_edges",
            sa.Column("id", sa.String(length=32), primary_key=True),
            sa.Column("source_ref", sa.String(length=255), nullable=False),
            sa.Column("target_ref", sa.String(length=255), nullable=False),
            sa.Column("relation", sa.String(length=64), nullable=False),
            sa.Column("attributes", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
    _ensure_index(inspector, "knowledge_edges", "ix_knowledge_edges_source_ref", ["source_ref"])
    _ensure_index(inspector, "knowledge_edges", "ix_knowledge_edges_target_ref", ["target_ref"])

    if "forecast_records" not in tables:
        op.create_table(
            "forecast_records",
            sa.Column("id", sa.String(length=32), primary_key=True),
            sa.Column("project_ref", sa.String(length=255), nullable=False),
            sa.Column("source", sa.String(length=32), nullable=False),
            sa.Column("delay_probability", sa.Float(), nullable=False),
            sa.Column("risk_trend", sa.Float(), nullable=False),
            sa.Column("forecast", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
    _ensure_index(inspector, "forecast_records", "ix_forecast_records_project_ref", ["project_ref"])

    if "simulation_runs" not in tables:
        op.create_table(
            "simulation_runs",
            sa.Column("id", sa.String(length=32), primary_key=True),
            sa.Column("project_ref", sa.String(length=255), nullable=False),
            sa.Column("scenario_name", sa.String(length=255), nullable=False),
            sa.Column("baseline", sa.JSON(), nullable=False),
            sa.Column("adjustments", sa.JSON(), nullable=False),
            sa.Column("outcome", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
    _ensure_index(inspector, "simulation_runs", "ix_simulation_runs_project_ref", ["project_ref"])

    if "world_model_states" not in tables:
        op.create_table(
            "world_model_states",
            sa.Column("id", sa.String(length=32), primary_key=True),
            sa.Column("project_ref", sa.String(length=255), nullable=False),
            sa.Column("state_kind", sa.String(length=64), nullable=False),
            sa.Column("latent_state", sa.JSON(), nullable=False),
            sa.Column("transitions", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
    _ensure_index(inspector, "world_model_states", "ix_world_model_states_project_ref", ["project_ref"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    tables = set(inspector.get_table_names())

    _drop_index_if_exists(inspector, "world_model_states", "ix_world_model_states_project_ref")
    if "world_model_states" in tables:
        op.drop_table("world_model_states")

    _drop_index_if_exists(inspector, "simulation_runs", "ix_simulation_runs_project_ref")
    if "simulation_runs" in tables:
        op.drop_table("simulation_runs")

    _drop_index_if_exists(inspector, "forecast_records", "ix_forecast_records_project_ref")
    if "forecast_records" in tables:
        op.drop_table("forecast_records")

    _drop_index_if_exists(inspector, "knowledge_edges", "ix_knowledge_edges_target_ref")
    _drop_index_if_exists(inspector, "knowledge_edges", "ix_knowledge_edges_source_ref")
    if "knowledge_edges" in tables:
        op.drop_table("knowledge_edges")

    _drop_index_if_exists(inspector, "knowledge_nodes", "ix_knowledge_nodes_external_ref")
    _drop_index_if_exists(inspector, "knowledge_nodes", "ix_knowledge_nodes_node_type")
    if "knowledge_nodes" in tables:
        op.drop_table("knowledge_nodes")


def _ensure_index(inspector, table_name: str, index_name: str, columns: list[str]) -> None:
    inspector = inspect(op.get_bind())
    if table_name not in inspector.get_table_names():
        return
    existing = {index["name"] for index in inspector.get_indexes(table_name)}
    if index_name not in existing:
        op.create_index(index_name, table_name, columns)


def _drop_index_if_exists(inspector, table_name: str, index_name: str) -> None:
    inspector = inspect(op.get_bind())
    if table_name not in inspector.get_table_names():
        return
    existing = {index["name"] for index in inspector.get_indexes(table_name)}
    if index_name in existing:
        op.drop_index(index_name, table_name=table_name)

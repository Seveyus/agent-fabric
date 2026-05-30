"""canonical organizational model

Revision ID: 0005_canonical_model
Revises: 0004_knowledge_layers
Create Date: 2026-05-31
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "0005_canonical_model"
down_revision = "0004_knowledge_layers"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    tables = set(inspector.get_table_names())

    _ensure_column(inspector, "jobs", sa.Column("project_ref", sa.String(length=255), nullable=True))
    _ensure_index(inspector, "jobs", "ix_jobs_project_ref", ["project_ref"])
    _ensure_column(inspector, "project_snapshots", sa.Column("project_ref", sa.String(length=255), nullable=True))
    _ensure_index(inspector, "project_snapshots", "ix_project_snapshots_project_ref", ["project_ref"])

    if "integrations" not in tables:
        op.create_table(
            "integrations",
            sa.Column("id", sa.String(length=32), primary_key=True),
            sa.Column("provider", sa.String(length=32), nullable=False),
            sa.Column("name", sa.String(length=255), nullable=False),
            sa.Column("base_url", sa.String(length=1024), nullable=False),
            sa.Column("auth_token", sa.Text(), nullable=False),
            sa.Column("config", sa.JSON(), nullable=False),
            sa.Column("status", sa.String(length=32), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
    _ensure_index(inspector, "integrations", "ix_integrations_provider", ["provider"])

    if "connector_sync_runs" not in tables:
        op.create_table(
            "connector_sync_runs",
            sa.Column("id", sa.String(length=32), primary_key=True),
            sa.Column("integration_id", sa.String(length=32), nullable=False),
            sa.Column("provider", sa.String(length=32), nullable=False),
            sa.Column("project_ref", sa.String(length=255), nullable=False),
            sa.Column("status", sa.String(length=32), nullable=False),
            sa.Column("raw_count", sa.Integer(), nullable=False),
            sa.Column("entity_count", sa.Integer(), nullable=False),
            sa.Column("relation_count", sa.Integer(), nullable=False),
            sa.Column("snapshot_count", sa.Integer(), nullable=False),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        )
    _ensure_index(inspector, "connector_sync_runs", "ix_connector_sync_runs_integration_id", ["integration_id"])
    _ensure_index(inspector, "connector_sync_runs", "ix_connector_sync_runs_project_ref", ["project_ref"])

    if "raw_external_events" not in tables:
        op.create_table(
            "raw_external_events",
            sa.Column("id", sa.String(length=32), primary_key=True),
            sa.Column("sync_run_id", sa.String(length=32), nullable=False),
            sa.Column("integration_id", sa.String(length=32), nullable=False),
            sa.Column("provider", sa.String(length=32), nullable=False),
            sa.Column("entity_type", sa.String(length=64), nullable=False),
            sa.Column("external_ref", sa.String(length=255), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
    _ensure_index(inspector, "raw_external_events", "ix_raw_external_events_sync_run_id", ["sync_run_id"])
    _ensure_index(inspector, "raw_external_events", "ix_raw_external_events_integration_id", ["integration_id"])
    _ensure_index(inspector, "raw_external_events", "ix_raw_external_events_external_ref", ["external_ref"])

    if "canonical_entities" not in tables:
        op.create_table(
            "canonical_entities",
            sa.Column("id", sa.String(length=32), primary_key=True),
            sa.Column("sync_run_id", sa.String(length=32), nullable=False),
            sa.Column("source_provider", sa.String(length=32), nullable=False),
            sa.Column("entity_type", sa.String(length=64), nullable=False),
            sa.Column("entity_ref", sa.String(length=255), nullable=False),
            sa.Column("workspace_ref", sa.String(length=255), nullable=True),
            sa.Column("project_ref", sa.String(length=255), nullable=True),
            sa.Column("name", sa.String(length=255), nullable=False),
            sa.Column("attributes", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
    _ensure_index(inspector, "canonical_entities", "ix_canonical_entities_sync_run_id", ["sync_run_id"])
    _ensure_index(inspector, "canonical_entities", "ix_canonical_entities_entity_type", ["entity_type"])
    _ensure_index(inspector, "canonical_entities", "ix_canonical_entities_entity_ref", ["entity_ref"])
    _ensure_index(inspector, "canonical_entities", "ix_canonical_entities_workspace_ref", ["workspace_ref"])
    _ensure_index(inspector, "canonical_entities", "ix_canonical_entities_project_ref", ["project_ref"])

    if "canonical_relations" not in tables:
        op.create_table(
            "canonical_relations",
            sa.Column("id", sa.String(length=32), primary_key=True),
            sa.Column("sync_run_id", sa.String(length=32), nullable=False),
            sa.Column("source_provider", sa.String(length=32), nullable=False),
            sa.Column("relation_type", sa.String(length=64), nullable=False),
            sa.Column("source_ref", sa.String(length=255), nullable=False),
            sa.Column("target_ref", sa.String(length=255), nullable=False),
            sa.Column("project_ref", sa.String(length=255), nullable=True),
            sa.Column("attributes", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
    _ensure_index(inspector, "canonical_relations", "ix_canonical_relations_sync_run_id", ["sync_run_id"])
    _ensure_index(inspector, "canonical_relations", "ix_canonical_relations_relation_type", ["relation_type"])
    _ensure_index(inspector, "canonical_relations", "ix_canonical_relations_source_ref", ["source_ref"])
    _ensure_index(inspector, "canonical_relations", "ix_canonical_relations_target_ref", ["target_ref"])
    _ensure_index(inspector, "canonical_relations", "ix_canonical_relations_project_ref", ["project_ref"])

    if "metric_snapshots" not in tables:
        op.create_table(
            "metric_snapshots",
            sa.Column("id", sa.String(length=32), primary_key=True),
            sa.Column("sync_run_id", sa.String(length=32), nullable=False),
            sa.Column("project_ref", sa.String(length=255), nullable=False),
            sa.Column("metric_source", sa.String(length=32), nullable=False),
            sa.Column("metric_name", sa.String(length=64), nullable=False),
            sa.Column("metric_value", sa.Float(), nullable=False),
            sa.Column("metric_unit", sa.String(length=32), nullable=True),
            sa.Column("dimensions", sa.JSON(), nullable=False),
            sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        )
    _ensure_index(inspector, "metric_snapshots", "ix_metric_snapshots_sync_run_id", ["sync_run_id"])
    _ensure_index(inspector, "metric_snapshots", "ix_metric_snapshots_project_ref", ["project_ref"])
    _ensure_index(inspector, "metric_snapshots", "ix_metric_snapshots_metric_source", ["metric_source"])
    _ensure_index(inspector, "metric_snapshots", "ix_metric_snapshots_metric_name", ["metric_name"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    tables = set(inspector.get_table_names())

    for table_name, indexes in [
        ("metric_snapshots", ["ix_metric_snapshots_metric_name", "ix_metric_snapshots_metric_source", "ix_metric_snapshots_project_ref", "ix_metric_snapshots_sync_run_id"]),
        ("canonical_relations", ["ix_canonical_relations_project_ref", "ix_canonical_relations_target_ref", "ix_canonical_relations_source_ref", "ix_canonical_relations_relation_type", "ix_canonical_relations_sync_run_id"]),
        ("canonical_entities", ["ix_canonical_entities_project_ref", "ix_canonical_entities_workspace_ref", "ix_canonical_entities_entity_ref", "ix_canonical_entities_entity_type", "ix_canonical_entities_sync_run_id"]),
        ("raw_external_events", ["ix_raw_external_events_external_ref", "ix_raw_external_events_integration_id", "ix_raw_external_events_sync_run_id"]),
        ("connector_sync_runs", ["ix_connector_sync_runs_project_ref", "ix_connector_sync_runs_integration_id"]),
        ("integrations", ["ix_integrations_provider"]),
    ]:
        for index_name in indexes:
            _drop_index_if_exists(inspector, table_name, index_name)
        if table_name in tables:
            op.drop_table(table_name)


def _ensure_column(inspector, table_name: str, column: sa.Column) -> None:
    inspector = inspect(op.get_bind())
    if table_name not in inspector.get_table_names():
        return
    existing = {item["name"] for item in inspector.get_columns(table_name)}
    if column.name not in existing:
        op.add_column(table_name, column)


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

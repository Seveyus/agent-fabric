from datetime import datetime, timezone
from uuid import uuid4

from core.connectors.github_connector import GitHubConnector
from core.connectors.jira_connector import JiraConnector
from core.connectors.notion_connector import NotionConnector
from db.models.canonical_entity import CanonicalEntity
from db.models.canonical_relation import CanonicalRelation
from db.models.connector_sync_run import ConnectorSyncRun
from db.models.integration import Integration
from db.models.metric_snapshot import MetricSnapshot
from db.models.raw_external_event import RawExternalEvent


CONNECTOR_MAP = {
    "github": GitHubConnector,
    "jira": JiraConnector,
    "notion": NotionConnector,
}


def run_connector_sync(db, integration: Integration, project_ref: str) -> ConnectorSyncRun:
    sync_run = ConnectorSyncRun(
        id=f"sync_{uuid4().hex[:12]}",
        integration_id=integration.id,
        provider=integration.provider,
        project_ref=project_ref,
        status="running",
    )
    db.add(sync_run)
    db.commit()
    db.refresh(sync_run)

    connector_cls = CONNECTOR_MAP.get(integration.provider)
    if connector_cls is None:
        sync_run.status = "failed"
        sync_run.error_message = f"Unsupported provider: {integration.provider}"
        sync_run.completed_at = datetime.now(timezone.utc)
        db.commit()
        raise ValueError(sync_run.error_message)

    try:
        payload = connector_cls(integration).sync(project_ref)
        _persist_raw_payloads(db, sync_run, integration, payload["raw_payload"], project_ref)
        _persist_entities(db, sync_run, integration, payload["entities"])
        _persist_relations(db, sync_run, integration, payload["relations"])
        _persist_metric_snapshots(db, sync_run, payload["metric_snapshots"])
        sync_run.raw_count = len(_flatten_raw_events(payload["raw_payload"], project_ref))
        sync_run.entity_count = len(payload["entities"])
        sync_run.relation_count = len(payload["relations"])
        sync_run.snapshot_count = len(payload["metric_snapshots"])
        sync_run.status = "completed"
        sync_run.completed_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(sync_run)
        return sync_run
    except Exception as exc:
        sync_run.status = "failed"
        sync_run.error_message = str(exc)
        sync_run.completed_at = datetime.now(timezone.utc)
        db.commit()
        raise


def _persist_raw_payloads(db, sync_run: ConnectorSyncRun, integration: Integration, raw_payload: dict, project_ref: str) -> None:
    for item in _flatten_raw_events(raw_payload, project_ref):
        db.add(
            RawExternalEvent(
                id=f"raw_{uuid4().hex[:12]}",
                sync_run_id=sync_run.id,
                integration_id=integration.id,
                provider=integration.provider,
                entity_type=item["entity_type"],
                external_ref=item["external_ref"],
                payload=item["payload"],
            )
        )
    db.commit()


def _persist_entities(db, sync_run: ConnectorSyncRun, integration: Integration, entities: list[dict]) -> None:
    for entity in entities:
        row = db.query(CanonicalEntity).filter(CanonicalEntity.entity_ref == entity["entity_ref"]).one_or_none()
        if row is None:
            row = CanonicalEntity(
                id=f"cent_{uuid4().hex[:12]}",
                sync_run_id=sync_run.id,
                source_provider=integration.provider,
                entity_type=entity["entity_type"],
                entity_ref=entity["entity_ref"],
                workspace_ref=f"workspace:{integration.provider}:{integration.name}",
                project_ref=entity.get("project_ref"),
                name=entity["name"],
                attributes=entity.get("attributes", {}),
            )
            db.add(row)
        else:
            row.sync_run_id = sync_run.id
            row.source_provider = integration.provider
            row.workspace_ref = f"workspace:{integration.provider}:{integration.name}"
            row.project_ref = entity.get("project_ref")
            row.name = entity["name"]
            row.attributes = entity.get("attributes", {})
    db.commit()


def _persist_relations(db, sync_run: ConnectorSyncRun, integration: Integration, relations: list[dict]) -> None:
    for relation in relations:
        existing = (
            db.query(CanonicalRelation)
            .filter(
                CanonicalRelation.source_ref == relation["source_ref"],
                CanonicalRelation.target_ref == relation["target_ref"],
                CanonicalRelation.relation_type == relation["relation_type"],
            )
            .one_or_none()
        )
        if existing is None:
            db.add(
                CanonicalRelation(
                    id=f"crel_{uuid4().hex[:12]}",
                    sync_run_id=sync_run.id,
                    source_provider=integration.provider,
                    relation_type=relation["relation_type"],
                    source_ref=relation["source_ref"],
                    target_ref=relation["target_ref"],
                    project_ref=relation.get("project_ref"),
                    attributes=relation.get("attributes", {}),
                )
            )
        else:
            existing.sync_run_id = sync_run.id
            existing.source_provider = integration.provider
            existing.project_ref = relation.get("project_ref")
            existing.attributes = relation.get("attributes", {})
    db.commit()


def _persist_metric_snapshots(db, sync_run: ConnectorSyncRun, snapshots: list[dict]) -> None:
    for snapshot in snapshots:
        db.add(
            MetricSnapshot(
                id=f"ms_{uuid4().hex[:12]}",
                sync_run_id=sync_run.id,
                project_ref=snapshot["project_ref"],
                metric_source=snapshot["metric_source"],
                metric_name=snapshot["metric_name"],
                metric_value=snapshot["metric_value"],
                metric_unit=snapshot.get("metric_unit"),
                dimensions=snapshot.get("dimensions", {}),
            )
        )
    db.commit()


def _flatten_raw_events(raw_payload: dict, project_ref: str) -> list[dict]:
    events = []
    for key, value in raw_payload.items():
        if isinstance(value, list):
            for idx, item in enumerate(value):
                external_ref = item.get("id") or item.get("key") or item.get("number") or f"{project_ref}:{key}:{idx}"
                events.append({"entity_type": key, "external_ref": str(external_ref), "payload": item})
        elif isinstance(value, dict):
            external_ref = value.get("id") or value.get("key") or value.get("full_name") or f"{project_ref}:{key}"
            events.append({"entity_type": key, "external_ref": str(external_ref), "payload": value})
    return events

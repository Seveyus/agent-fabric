from uuid import uuid4

from db.models.knowledge_edge import KnowledgeEdge
from db.models.knowledge_node import KnowledgeNode


def upsert_node(db, node_type: str, external_ref: str, title: str, attributes: dict) -> KnowledgeNode:
    row = (
        db.query(KnowledgeNode)
        .filter(KnowledgeNode.node_type == node_type, KnowledgeNode.external_ref == external_ref)
        .one_or_none()
    )
    if row is None:
        row = KnowledgeNode(
            id=f"node_{uuid4().hex[:12]}",
            node_type=node_type,
            external_ref=external_ref,
            title=title,
            attributes=attributes,
        )
        db.add(row)
    else:
        row.title = title
        row.attributes = attributes
    db.flush()
    return row


def ensure_edge(db, source_ref: str, target_ref: str, relation: str, attributes: dict | None = None) -> None:
    row = (
        db.query(KnowledgeEdge)
        .filter(
            KnowledgeEdge.source_ref == source_ref,
            KnowledgeEdge.target_ref == target_ref,
            KnowledgeEdge.relation == relation,
        )
        .one_or_none()
    )
    if row is None:
        db.add(
            KnowledgeEdge(
                id=f"edge_{uuid4().hex[:12]}",
                source_ref=source_ref,
                target_ref=target_ref,
                relation=relation,
                attributes=attributes or {},
            )
        )
    else:
        row.attributes = attributes or {}
    db.flush()


def ingest_telemetry_snapshot_into_graph(db, snapshot) -> None:
    project_ref = snapshot.project_ref
    project_node = upsert_node(
        db,
        "project",
        project_ref,
        project_ref,
        {
            "provider": snapshot.provider,
            "risk_score": snapshot.risk_score,
            "risk_level": snapshot.risk_level,
            **snapshot.metrics,
        },
    )
    provider_ref = f"provider:{snapshot.provider}:{snapshot.connection_id}"
    provider_node = upsert_node(
        db,
        "integration",
        provider_ref,
        f"{snapshot.provider.upper()} connection",
        {"connection_id": snapshot.connection_id, "provider": snapshot.provider},
    )
    ensure_edge(db, provider_node.external_ref, project_node.external_ref, "observes")

    for reason in snapshot.evidence.get("reasons", []):
        signal_ref = f"signal:{snapshot.id}:{abs(hash(reason)) % 10_000_000}"
        signal_node = upsert_node(
            db,
            "risk_signal",
            signal_ref,
            reason[:120],
            {"snapshot_id": snapshot.id, "provider": snapshot.provider, "reason": reason},
        )
        ensure_edge(db, project_node.external_ref, signal_node.external_ref, "has_risk_signal")

    db.commit()


def ingest_document_snapshot_into_graph(db, run_id: str, snapshot: dict, findings: list[dict]) -> None:
    project_ref = f"run:{run_id}"
    run_node = upsert_node(
        db,
        "document_run",
        project_ref,
        f"Document run {run_id}",
        snapshot,
    )
    for finding in findings:
        finding_ref = f"finding:{run_id}:{finding['kind']}"
        finding_node = upsert_node(
            db,
            "finding",
            finding_ref,
            finding["title"],
            {
                "kind": finding["kind"],
                "severity": finding["severity"],
                "summary": finding["summary"],
                "confidence": finding["confidence"],
            },
        )
        ensure_edge(db, run_node.external_ref, finding_node.external_ref, "produced")
    db.commit()

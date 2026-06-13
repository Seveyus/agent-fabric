"""
Temporal knowledge graph service — append-only causal edge semantics.

Key design decision: edges are NEVER overwritten.
When a relationship changes, we close the old edge (valid_to = now) and
insert a new one. This preserves the full causal history.

Causal edge types:
    triggered_by    event/entity A was caused by event/entity B
    constrained_by  entity A is limited by entity B (technical/legal/resource)
    leads_to        entity A causally produces entity B
    overruled_by    decision A was reversed by decision B
    blocks          standard blocking dependency
    depends_on      softer dependency (non-blocking)
    observes        monitoring/integration relationship
    modifies        a PR/commit modifies a file/component
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from db.models.canonical_entity import CanonicalEntity
from db.models.canonical_relation import CanonicalRelation
from db.models.knowledge_edge import KnowledgeEdge
from db.models.knowledge_node import KnowledgeNode

log = logging.getLogger(__name__)

CAUSAL_RELATION_TYPES = {
    "triggered_by",
    "constrained_by",
    "leads_to",
    "overruled_by",
    "blocks",
    "depends_on",
    "observes",
    "modifies",
}


# ------------------------------------------------------------------ #
# KnowledgeNode / KnowledgeEdge (runtime graph — connector syncs)     #
# ------------------------------------------------------------------ #

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


def append_edge(
    db,
    source_ref: str,
    target_ref: str,
    relation: str,
    attributes: dict | None = None,
    confidence: float = 1.0,
) -> KnowledgeEdge:
    """
    Append-only edge insert: never overwrites an existing edge.
    Closes the previous active edge for the same (source, target, relation)
    triplet before inserting a new one.
    """
    now = datetime.now(timezone.utc)

    # Close any currently-active edge for this triplet
    active = (
        db.query(KnowledgeEdge)
        .filter(
            KnowledgeEdge.source_ref == source_ref,
            KnowledgeEdge.target_ref == target_ref,
            KnowledgeEdge.relation == relation,
            KnowledgeEdge.valid_to.is_(None),
        )
        .one_or_none()
    )
    if active:
        active.valid_to = now
        db.flush()

    new_edge = KnowledgeEdge(
        id=f"edge_{uuid4().hex[:12]}",
        source_ref=source_ref,
        target_ref=target_ref,
        relation=relation,
        attributes=attributes or {},
        confidence=confidence,
        valid_from=now,
        valid_to=None,
    )
    db.add(new_edge)
    db.flush()
    return new_edge


def ensure_edge(db, source_ref: str, target_ref: str, relation: str, attributes: dict | None = None) -> None:
    """
    Idempotent edge: only inserts if no active edge exists for this triplet.
    Use for non-causal structural edges (observes, etc.).
    """
    active = (
        db.query(KnowledgeEdge)
        .filter(
            KnowledgeEdge.source_ref == source_ref,
            KnowledgeEdge.target_ref == target_ref,
            KnowledgeEdge.relation == relation,
            KnowledgeEdge.valid_to.is_(None),
        )
        .one_or_none()
    )
    if active is None:
        db.add(
            KnowledgeEdge(
                id=f"edge_{uuid4().hex[:12]}",
                source_ref=source_ref,
                target_ref=target_ref,
                relation=relation,
                attributes=attributes or {},
                confidence=1.0,
                valid_from=datetime.now(timezone.utc),
                valid_to=None,
            )
        )
        db.flush()


def ingest_extraction_result(db, extraction: dict, project_ref: str, source_ref: str) -> None:
    """
    Persist a causal decision extracted by ExtractorAgent into the knowledge graph.
    """
    if not extraction.get("has_decision") or not extraction.get("decision"):
        return

    decision_ref = f"decision:{source_ref}:{abs(hash(extraction['decision'])) % 10_000_000}"
    decision_node = upsert_node(
        db,
        "decision",
        decision_ref,
        extraction["decision"][:120],
        {
            "rationale": extraction.get("rationale", ""),
            "alternatives_rejected": extraction.get("alternatives_rejected", []),
            "causal_type": extraction.get("causal_type", "general"),
            "reversed_by": extraction.get("reversed_by"),
            "confidence": extraction.get("confidence", 0.5),
            "project_ref": project_ref,
            "source_ref": source_ref,
        },
    )

    # Causal edge from the source artifact to the decision
    causal_type = extraction.get("causal_type", "triggered_by")
    if causal_type not in CAUSAL_RELATION_TYPES:
        causal_type = "triggered_by"

    append_edge(
        db,
        source_ref=source_ref,
        target_ref=decision_node.external_ref,
        relation=causal_type,
        attributes={"confidence": extraction.get("confidence", 0.5)},
        confidence=extraction.get("confidence", 0.5),
    )

    # If this decision was reversed by another, mark the reversal
    reversed_by = extraction.get("reversed_by")
    if reversed_by:
        append_edge(
            db,
            source_ref=decision_node.external_ref,
            target_ref=reversed_by,
            relation="overruled_by",
            attributes={"note": "decision reversed"},
            confidence=0.8,
        )

    db.commit()


# ------------------------------------------------------------------ #
# CanonicalEntity / CanonicalRelation (cross-source dedup layer)      #
# ------------------------------------------------------------------ #

def upsert_canonical_entity(
    db,
    sync_run_id: str,
    source_provider: str,
    entity_type: str,
    entity_ref: str,
    name: str,
    attributes: dict,
    project_ref: str | None = None,
    workspace_ref: str | None = None,
) -> CanonicalEntity:
    row = (
        db.query(CanonicalEntity)
        .filter(CanonicalEntity.entity_ref == entity_ref)
        .one_or_none()
    )
    if row is None:
        row = CanonicalEntity(
            id=f"ce_{uuid4().hex[:12]}",
            sync_run_id=sync_run_id,
            source_provider=source_provider,
            entity_type=entity_type,
            entity_ref=entity_ref,
            name=name,
            attributes=attributes,
            project_ref=project_ref,
            workspace_ref=workspace_ref,
        )
        db.add(row)
    else:
        row.sync_run_id = sync_run_id
        row.name = name
        row.attributes = attributes
    db.flush()
    return row


def append_causal_relation(
    db,
    sync_run_id: str,
    source_provider: str,
    relation_type: str,
    source_ref: str,
    target_ref: str,
    project_ref: str | None = None,
    attributes: dict | None = None,
    confidence: float = 1.0,
) -> CanonicalRelation:
    """
    Append-only causal relation insert.
    Closes the previous active relation for this triplet before inserting.
    """
    now = datetime.now(timezone.utc)

    active = (
        db.query(CanonicalRelation)
        .filter(
            CanonicalRelation.source_ref == source_ref,
            CanonicalRelation.target_ref == target_ref,
            CanonicalRelation.relation_type == relation_type,
            CanonicalRelation.valid_to.is_(None),
        )
        .one_or_none()
    )
    if active:
        active.valid_to = now
        db.flush()

    row = CanonicalRelation(
        id=f"cr_{uuid4().hex[:12]}",
        sync_run_id=sync_run_id,
        source_provider=source_provider,
        relation_type=relation_type,
        source_ref=source_ref,
        target_ref=target_ref,
        project_ref=project_ref,
        attributes=attributes or {},
        confidence=confidence,
        valid_from=now,
        valid_to=None,
    )
    db.add(row)
    db.flush()
    return row


def get_causal_chain(
    db,
    entity_ref: str,
    depth: int = 2,
    active_only: bool = True,
) -> list[dict]:
    """
    Traverse the causal graph from entity_ref up to `depth` hops.
    Returns list of edge dicts for the pre-flight endpoint.
    """
    visited = set()
    results = []
    _traverse(db, entity_ref, depth, active_only, visited, results)
    return results


def _traverse(db, ref: str, depth: int, active_only: bool, visited: set, results: list) -> None:
    if depth <= 0 or ref in visited:
        return
    visited.add(ref)

    query = db.query(CanonicalRelation).filter(CanonicalRelation.source_ref == ref)
    if active_only:
        query = query.filter(CanonicalRelation.valid_to.is_(None))

    for rel in query.limit(20).all():
        results.append({
            "source_ref": rel.source_ref,
            "target_ref": rel.target_ref,
            "relation_type": rel.relation_type,
            "confidence": rel.confidence,
            "valid_from": rel.valid_from.isoformat() if rel.valid_from else None,
            "valid_to": rel.valid_to.isoformat() if rel.valid_to else None,
            "attributes": rel.attributes,
        })
        _traverse(db, rel.target_ref, depth - 1, active_only, visited, results)

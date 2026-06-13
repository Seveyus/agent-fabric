"""
Pre-flight risk endpoint — the core product API.

Answers: "Before an AI agent (or a developer) modifies artifact X,
what are the predicted downstream consequences?"

This is the MCP-compatible endpoint that Cursor, Claude Code, and other
agents call before executing a change.

Endpoints:
    POST /preflight          — risk assessment for a planned change
    GET  /preflight/context/{entity_ref}  — causal graph context for an entity
    GET  /preflight/history/{entity_ref}  — temporal history of an entity's edges
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from apps.api.deps import get_db
from core.services.knowledge_graph_service import get_causal_chain
from db.models.canonical_entity import CanonicalEntity
from db.models.canonical_relation import CanonicalRelation

log = logging.getLogger(__name__)
router = APIRouter(prefix="/preflight", tags=["preflight"])


# ------------------------------------------------------------------ #
# Request / Response schemas                                           #
# ------------------------------------------------------------------ #

class PreflightRequest(BaseModel):
    action: str = Field(..., description="What the agent intends to do", examples=["modify", "delete", "create"])
    target_ref: str = Field(..., description="Entity reference (file path, PR ref, ticket ID, etc.)")
    project_ref: str | None = Field(None, description="Optional project scope")
    context: dict = Field(default_factory=dict, description="Additional context from the agent")


class CausalConsequence(BaseModel):
    entity_ref: str
    entity_type: str
    entity_name: str
    relation_type: str
    probability: float = Field(ge=0.0, le=1.0)
    reason: str
    depth: int


class PreflightResponse(BaseModel):
    target_ref: str
    action: str
    risk_level: str              # low | medium | high | critical
    risk_score: float            # 0.0 – 1.0
    consequences: list[CausalConsequence]
    decision_history: list[dict] # relevant past decisions from the causal graph
    recommendation: str
    confidence: float


class ContextResponse(BaseModel):
    entity_ref: str
    entity_name: str | None
    causal_edges: list[dict]
    decision_nodes: list[dict]


# ------------------------------------------------------------------ #
# Routes                                                               #
# ------------------------------------------------------------------ #

@router.post("", response_model=PreflightResponse)
def assess_preflight(
    req: PreflightRequest,
    db: Annotated[Session, Depends(get_db)],
) -> PreflightResponse:
    """
    Core pre-flight assessment.

    Given a planned action on a target entity, returns:
    - downstream consequences with probabilities
    - relevant decision history
    - overall risk score
    - recommended action
    """
    # Fetch the target entity
    entity = (
        db.query(CanonicalEntity)
        .filter(CanonicalEntity.entity_ref == req.target_ref)
        .first()
    )

    # Traverse the causal graph
    causal_chain = get_causal_chain(db, req.target_ref, depth=2, active_only=True)

    # Build consequences from causal chain
    consequences: list[CausalConsequence] = []
    for edge in causal_chain:
        downstream = (
            db.query(CanonicalEntity)
            .filter(CanonicalEntity.entity_ref == edge["target_ref"])
            .first()
        )
        if downstream is None:
            continue

        prob = _edge_to_probability(edge)
        reason = _build_reason(edge, downstream)
        consequences.append(
            CausalConsequence(
                entity_ref=downstream.entity_ref,
                entity_type=downstream.entity_type,
                entity_name=downstream.name,
                relation_type=edge["relation_type"],
                probability=prob,
                reason=reason,
                depth=1,  # simplified — depth tracking to be added
            )
        )

    # Fetch relevant decisions involving this entity
    decision_history = _get_decision_history(db, req.target_ref, req.project_ref)

    # Compute risk score
    risk_score, risk_level = _compute_risk(consequences, decision_history, req.action)

    recommendation = _build_recommendation(risk_level, consequences, decision_history)

    mean_confidence = (
        sum(c.probability for c in consequences) / len(consequences)
        if consequences else 0.5
    )

    return PreflightResponse(
        target_ref=req.target_ref,
        action=req.action,
        risk_level=risk_level,
        risk_score=round(risk_score, 3),
        consequences=sorted(consequences, key=lambda c: c.probability, reverse=True)[:10],
        decision_history=decision_history[:5],
        recommendation=recommendation,
        confidence=round(mean_confidence, 3),
    )


@router.get("/context/{entity_ref}", response_model=ContextResponse)
def get_entity_context(
    entity_ref: str,
    db: Annotated[Session, Depends(get_db)],
) -> ContextResponse:
    """
    Return the full causal context for an entity.
    Designed to be injected directly into AI agent system prompts (MCP use).
    """
    entity = db.query(CanonicalEntity).filter(CanonicalEntity.entity_ref == entity_ref).first()
    causal_edges = get_causal_chain(db, entity_ref, depth=2, active_only=True)

    # Fetch decision nodes in this causal subgraph
    decision_refs = {
        e["target_ref"] for e in causal_edges
        if e["relation_type"] in {"triggered_by", "constrained_by", "overruled_by"}
    }
    decision_nodes = []
    for ref in decision_refs:
        node = db.query(CanonicalEntity).filter(
            CanonicalEntity.entity_ref == ref,
            CanonicalEntity.entity_type == "decision",
        ).first()
        if node:
            decision_nodes.append({
                "ref": node.entity_ref,
                "decision": node.name,
                "rationale": node.attributes.get("rationale", ""),
                "causal_type": node.attributes.get("causal_type", "general"),
                "confidence": node.attributes.get("confidence", 0.5),
            })

    return ContextResponse(
        entity_ref=entity_ref,
        entity_name=entity.name if entity else None,
        causal_edges=causal_edges,
        decision_nodes=decision_nodes,
    )


@router.get("/history/{entity_ref}")
def get_edge_history(
    entity_ref: str,
    db: Annotated[Session, Depends(get_db)],
) -> list[dict]:
    """
    Return the full temporal history of edges involving this entity,
    including closed (valid_to != null) edges.
    """
    edges = (
        db.query(CanonicalRelation)
        .filter(CanonicalRelation.source_ref == entity_ref)
        .order_by(CanonicalRelation.valid_from.desc())
        .limit(50)
        .all()
    )
    return [
        {
            "source_ref": e.source_ref,
            "target_ref": e.target_ref,
            "relation_type": e.relation_type,
            "confidence": e.confidence,
            "valid_from": e.valid_from.isoformat() if e.valid_from else None,
            "valid_to": e.valid_to.isoformat() if e.valid_to else None,
            "is_active": e.valid_to is None,
            "attributes": e.attributes,
        }
        for e in edges
    ]


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _edge_to_probability(edge: dict) -> float:
    base = edge.get("confidence", 0.5)
    # Causal edges carry higher probability of impact
    multipliers = {
        "triggered_by": 0.9,
        "constrained_by": 0.8,
        "leads_to": 0.85,
        "overruled_by": 0.7,
        "blocks": 0.95,
        "depends_on": 0.7,
        "modifies": 0.8,
    }
    return min(1.0, base * multipliers.get(edge["relation_type"], 0.6))


def _build_reason(edge: dict, downstream: "CanonicalEntity") -> str:
    rel = edge["relation_type"]
    name = downstream.name
    reasons = {
        "triggered_by": f"{name} was originally triggered by this entity",
        "constrained_by": f"{name} is constrained by this — changing it may violate the constraint",
        "leads_to": f"This causally produces {name}",
        "overruled_by": f"{name} previously overruled a decision here",
        "blocks": f"{name} is blocked by this entity",
        "depends_on": f"{name} depends on this entity",
        "modifies": f"This entity modifies {name}",
    }
    return reasons.get(rel, f"{name} is related via {rel}")


def _get_decision_history(db, entity_ref: str, project_ref: str | None) -> list[dict]:
    query = db.query(CanonicalEntity).filter(CanonicalEntity.entity_type == "decision")
    if project_ref:
        query = query.filter(CanonicalEntity.project_ref == project_ref)
    decisions = query.order_by(CanonicalEntity.created_at.desc()).limit(20).all()

    relevant = []
    for d in decisions:
        src = d.attributes.get("source_ref", "")
        if entity_ref in src or entity_ref in d.attributes.get("project_ref", ""):
            relevant.append({
                "ref": d.entity_ref,
                "decision": d.name,
                "rationale": d.attributes.get("rationale", ""),
                "causal_type": d.attributes.get("causal_type"),
                "confidence": d.attributes.get("confidence", 0.5),
                "reversed_by": d.attributes.get("reversed_by"),
            })
    return relevant


def _compute_risk(
    consequences: list[CausalConsequence],
    decisions: list[dict],
    action: str,
) -> tuple[float, str]:
    if not consequences:
        score = 0.1
    else:
        high_prob = [c for c in consequences if c.probability >= 0.7]
        score = min(0.99, 0.1 + len(high_prob) * 0.15 + len(consequences) * 0.03)

    # Reversals in history are a red flag
    reversals = sum(1 for d in decisions if d.get("reversed_by"))
    score = min(0.99, score + reversals * 0.1)

    # Delete actions are inherently riskier
    if action == "delete":
        score = min(0.99, score + 0.15)

    if score >= 0.7:
        return score, "critical" if score >= 0.85 else "high"
    elif score >= 0.4:
        return score, "medium"
    return score, "low"


def _build_recommendation(
    risk_level: str,
    consequences: list[CausalConsequence],
    decisions: list[dict],
) -> str:
    if risk_level in {"critical", "high"}:
        top = consequences[0].entity_name if consequences else "downstream entities"
        return (
            f"Require human review before proceeding — high-probability impact on {top}. "
            f"Check decision history for prior reversals."
        )
    if risk_level == "medium":
        return "Proceed with caution. Notify relevant stakeholders of potential impact."
    return "Low risk. Safe to proceed."

from collections import defaultdict

from db.models.canonical_relation import CanonicalRelation
from db.models.metric_snapshot import MetricSnapshot
from db.models.project_snapshot import ProjectSnapshot


def compute_project_risk_score(db, project_ref: str) -> dict:
    metric_rows = (
        db.query(MetricSnapshot)
        .filter(MetricSnapshot.project_ref == project_ref)
        .order_by(MetricSnapshot.captured_at.desc())
        .all()
    )
    latest_by_name = {}
    history_by_name = defaultdict(list)
    for row in metric_rows:
        if row.metric_name not in latest_by_name:
            latest_by_name[row.metric_name] = row
        history_by_name[row.metric_name].append(row.metric_value)

    doc_snapshot = (
        db.query(ProjectSnapshot)
        .filter(ProjectSnapshot.project_ref == project_ref)
        .order_by(ProjectSnapshot.created_at.desc())
        .first()
    )
    relations = db.query(CanonicalRelation).filter(CanonicalRelation.project_ref == project_ref).all()

    score = 0.08
    reasons = []
    actions = []
    evidence = []

    blocked = _metric_value(latest_by_name, "blocked_tickets")
    stale_prs = _metric_value(latest_by_name, "stale_prs")
    release_recency = _metric_value(latest_by_name, "release_recency_days")
    unassigned = _metric_value(latest_by_name, "unassigned_tickets")
    backlog = _metric_value(latest_by_name, "backlog_size")
    commit_velocity = _metric_value(latest_by_name, "commit_velocity_7d")
    recent_decisions = _metric_value(latest_by_name, "recent_decisions")
    stale_docs = _metric_value(latest_by_name, "stale_docs")
    undocumented = _metric_value(latest_by_name, "undocumented_projects")

    if blocked >= 5:
        score += 0.22
        reasons.append(f"{int(blocked)} blocked tickets")
        actions.append("Assign owners to unassigned blockers")
        evidence.append({"source": "jira", "metric": "blocked_tickets", "value": blocked})
    if stale_prs >= 2:
        score += 0.18
        reasons.append(f"{int(stale_prs)} stale PRs")
        actions.append("Review stale PRs")
        evidence.append({"source": "github", "metric": "stale_prs", "value": stale_prs})
    if commit_velocity <= 1:
        score += 0.16
        reasons.append("No merged PR or almost no commit activity in recent days")
        actions.append("Confirm delivery date with project owner")
        evidence.append({"source": "github", "metric": "commit_velocity_7d", "value": commit_velocity})
    if unassigned >= 3:
        score += 0.1
        reasons.append(f"{int(unassigned)} unassigned tickets")
        actions.append("Assign owners to uncovered work")
    if release_recency >= 21:
        score += 0.08
        reasons.append(f"No recent release in {int(release_recency)} days")
    if stale_docs >= 3 or undocumented >= 1:
        score += 0.08
        reasons.append("Documentation is stale or incomplete")
        actions.append("Refresh delivery docs and decisions")
    if recent_decisions >= 3:
        score += 0.06
        reasons.append(f"{int(recent_decisions)} recent decisions mention this project")
    if backlog and len(history_by_name["backlog_size"]) >= 2:
        newest = history_by_name["backlog_size"][0]
        oldest = history_by_name["backlog_size"][-1]
        if oldest > 0:
            growth = (newest - oldest) / oldest
            if growth >= 0.4:
                score += 0.12
                reasons.append(f"Backlog increased by {int(growth * 100)}% over recent syncs")
                evidence.append({"source": "jira", "metric": "backlog_growth", "value": round(growth, 2)})

    graph_blockers = sum(1 for row in relations if row.relation_type in {"blocks", "depends_on"})
    if graph_blockers >= 3:
        score += 0.08
        reasons.append(f"{graph_blockers} blocking graph relationships detected")

    if doc_snapshot:
        if doc_snapshot.blockers_detected > 0:
            score += 0.08
            reasons.append(f"{doc_snapshot.blockers_detected} blockers found in project documents")
        if doc_snapshot.owner_coverage_ratio < 0.7:
            score += 0.08
            reasons.append("Documented work has low ownership coverage")

    score = min(0.99, round(score, 2))
    risk_level = "high" if score >= 0.7 else "medium" if score >= 0.4 else "low"

    if not reasons:
        reasons.append("No major cross-tool risk signal detected")
    if not actions:
        actions.append("Continue monitoring current delivery cadence")

    return {
        "project_ref": project_ref,
        "risk_score": score,
        "risk_level": risk_level,
        "top_reasons": reasons[:4],
        "recommended_actions": list(dict.fromkeys(actions))[:4],
        "evidence": evidence[:8],
    }


def _metric_value(latest_by_name: dict, metric_name: str) -> float:
    row = latest_by_name.get(metric_name)
    return float(row.metric_value) if row else 0.0

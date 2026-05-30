from uuid import uuid4

from core.services.knowledge_graph_service import ingest_telemetry_snapshot_into_graph
from core.integrations.github_client import GitHubTelemetryClient
from core.integrations.jira_client import JiraTelemetryClient
from db.models.integration_connection import IntegrationConnection
from db.models.telemetry_snapshot import TelemetrySnapshot


def collect_telemetry_snapshot(db, connection: IntegrationConnection, project_ref: str) -> TelemetrySnapshot:
    if connection.provider == "github":
        metrics = GitHubTelemetryClient(connection.base_url, connection.auth_token).collect_repo_metrics(project_ref)
        risk_score, risk_level, evidence = score_github_metrics(metrics)
    elif connection.provider == "jira":
        metrics = JiraTelemetryClient(connection.base_url, connection.auth_token).collect_project_metrics(project_ref)
        risk_score, risk_level, evidence = score_jira_metrics(metrics)
    else:
        raise ValueError(f"Unsupported provider: {connection.provider}")

    snapshot = TelemetrySnapshot(
        id=f"telemetry_{uuid4().hex[:12]}",
        connection_id=connection.id,
        provider=connection.provider,
        project_ref=project_ref,
        risk_score=risk_score,
        risk_level=risk_level,
        metrics=metrics,
        evidence=evidence,
    )
    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)
    ingest_telemetry_snapshot_into_graph(db, snapshot)
    return snapshot


def score_github_metrics(metrics: dict) -> tuple[float, str, dict]:
    score = 5.0
    evidence = []

    if metrics["last_commit_age_days"] >= 7:
        score += 30.0
        evidence.append(f"No recent code activity for {metrics['last_commit_age_days']} days")
    if metrics["stale_pull_requests"] >= 2:
        score += 20.0
        evidence.append(f"{metrics['stale_pull_requests']} pull requests have been stale for over 7 days")
    if metrics["open_pull_requests"] >= 8:
        score += 15.0
        evidence.append(f"{metrics['open_pull_requests']} pull requests are currently open")
    if metrics["active_contributors_30d"] <= 1:
        score += 15.0
        evidence.append("Delivery activity is concentrated on one contributor or fewer")
    if metrics["commits_last_7d"] == 0:
        score += 15.0
        evidence.append("No commits were pushed in the last 7 days")

    return finalize_score(score, evidence, metrics)


def score_jira_metrics(metrics: dict) -> tuple[float, str, dict]:
    score = 5.0
    evidence = []

    if metrics["blocked_tickets"] >= 3:
        score += 30.0
        evidence.append(f"{metrics['blocked_tickets']} tickets are explicitly blocked")
    if metrics["stale_tickets"] >= 5:
        score += 20.0
        evidence.append(f"{metrics['stale_tickets']} tickets have not moved for over 14 days")
    if metrics["unassigned_tickets"] >= 5:
        score += 20.0
        evidence.append(f"{metrics['unassigned_tickets']} tickets are unassigned")
    if metrics["overdue_tickets"] >= 3:
        score += 20.0
        evidence.append(f"{metrics['overdue_tickets']} tickets are overdue")
    if metrics["open_tickets"] >= 30:
        score += 10.0
        evidence.append(f"{metrics['open_tickets']} open tickets remain in the backlog")

    return finalize_score(score, evidence, metrics)


def finalize_score(score: float, evidence_items: list[str], metrics: dict) -> tuple[float, str, dict]:
    score = min(100.0, round(score, 1))
    if score >= 70:
        risk_level = "high"
    elif score >= 40:
        risk_level = "medium"
    else:
        risk_level = "low"
    return score, risk_level, {"reasons": evidence_items, "metrics_summary": metrics}

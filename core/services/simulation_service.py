from copy import deepcopy
from uuid import uuid4

from core.services.project_risk_score_service import compute_project_risk_score
from db.models.simulation_run import SimulationRun
from db.models.metric_snapshot import MetricSnapshot


def run_project_simulation(db, project_ref: str, scenario_name: str, adjustments: dict) -> SimulationRun:
    metric_rows = (
        db.query(MetricSnapshot)
        .filter(MetricSnapshot.project_ref == project_ref)
        .order_by(MetricSnapshot.captured_at.desc())
        .all()
    )
    if not metric_rows:
        raise ValueError(f"No metric snapshot found for project '{project_ref}'")

    baseline_metrics = {}
    source_counts = {}
    for row in metric_rows:
        baseline_metrics.setdefault(row.metric_name, row.metric_value)
        source_counts[row.metric_source] = source_counts.get(row.metric_source, 0) + 1
    primary_source = max(source_counts, key=source_counts.get)
    simulated_metrics = apply_adjustments(baseline_metrics, adjustments)
    baseline_score = compute_project_risk_score(db, project_ref)
    score, risk_level, evidence = rescore_metrics(primary_source, simulated_metrics)
    recommendations = build_recommendations(primary_source, baseline_metrics, simulated_metrics)

    row = SimulationRun(
        id=f"sim_{uuid4().hex[:12]}",
        project_ref=project_ref,
        scenario_name=scenario_name,
        baseline=baseline_metrics,
        adjustments=adjustments,
        outcome={
            "risk_score": score,
            "risk_level": risk_level,
            "baseline_risk_score": baseline_score["risk_score"],
            "evidence": evidence,
            "recommendations": recommendations,
            "simulated_metrics": simulated_metrics,
        },
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def apply_adjustments(metrics: dict, adjustments: dict) -> dict:
    simulated = deepcopy(metrics)
    for key, value in adjustments.items():
        current = simulated.get(key)
        if isinstance(current, (int, float)) and isinstance(value, (int, float)):
            simulated[key] = max(0, current + value)
        else:
            simulated[key] = value
    return simulated


def rescore_metrics(provider: str, metrics: dict) -> tuple[float, str, dict]:
    if provider == "github":
        score = 5.0
        reasons = []
        last_commit_age = _github_metric(metrics, "last_commit_age_days", default=999.0)
        stale_prs = _github_metric(metrics, "stale_prs", legacy_key="stale_pull_requests")
        open_prs = _github_metric(metrics, "open_pull_requests")
        contributors = _github_metric(metrics, "active_contributors_30d")
        commit_velocity = _github_metric(metrics, "commit_velocity_7d", legacy_key="commits_last_7d")

        if last_commit_age >= 7:
            score += 30.0
            reasons.append(f"No recent code activity for {last_commit_age} days")
        if stale_prs >= 2:
            score += 20.0
            reasons.append(f"{stale_prs} pull requests remain stale")
        if open_prs >= 8:
            score += 15.0
            reasons.append(f"{open_prs} pull requests are open")
        if contributors <= 1:
            score += 15.0
            reasons.append("Contributor concentration remains high")
        if commit_velocity <= 1:
            score += 15.0
            reasons.append("Very low code throughput in the last 7 days")
        return finalize_score(score, reasons, metrics)

    score = 5.0
    reasons = []
    if metrics.get("blocked_tickets", 0) >= 3:
        score += 30.0
        reasons.append(f"{metrics['blocked_tickets']} tickets are blocked")
    if metrics.get("stale_tickets", 0) >= 5:
        score += 20.0
        reasons.append(f"{metrics['stale_tickets']} tickets are stale")
    if metrics.get("unassigned_tickets", 0) >= 5:
        score += 20.0
        reasons.append(f"{metrics['unassigned_tickets']} tickets are unassigned")
    if metrics.get("overdue_tickets", 0) >= 3:
        score += 20.0
        reasons.append(f"{metrics['overdue_tickets']} tickets are overdue")
    if metrics.get("open_tickets", 0) >= 30:
        score += 10.0
        reasons.append(f"{metrics['open_tickets']} tickets remain open")
    return finalize_score(score, reasons, metrics)


def build_recommendations(provider: str, baseline: dict, simulated: dict) -> list[str]:
    recommendations = []
    if provider == "github":
        if _github_metric(simulated, "active_contributors_30d") > _github_metric(baseline, "active_contributors_30d"):
            recommendations.append("Adding contributors reduces delivery concentration risk.")
        if _github_metric(simulated, "stale_prs", legacy_key="stale_pull_requests") < _github_metric(
            baseline, "stale_prs", legacy_key="stale_pull_requests"
        ):
            recommendations.append("Reducing stale PRs materially improves merge flow.")
        if _github_metric(simulated, "commit_velocity_7d", legacy_key="commits_last_7d") > _github_metric(
            baseline, "commit_velocity_7d", legacy_key="commits_last_7d"
        ):
            recommendations.append("Increasing weekly commit activity lowers inactivity risk.")
    else:
        if simulated.get("blocked_tickets", 0) < baseline.get("blocked_tickets", 0):
            recommendations.append("Unblocking tickets is the highest leverage intervention.")
        if simulated.get("unassigned_tickets", 0) < baseline.get("unassigned_tickets", 0):
            recommendations.append("Assigning orphan tickets improves ownership coverage.")
        if simulated.get("stale_tickets", 0) < baseline.get("stale_tickets", 0):
            recommendations.append("Refreshing stale tickets should improve execution cadence.")
    if not recommendations:
        recommendations.append("This scenario changes little; add stronger interventions to move the risk profile.")
    return recommendations


def _github_metric(metrics: dict, canonical_key: str, legacy_key: str | None = None, default: float = 0.0) -> float:
    if canonical_key in metrics:
        return float(metrics[canonical_key])
    if legacy_key and legacy_key in metrics:
        return float(metrics[legacy_key])
    return default


def finalize_score(score: float, reasons: list[str], metrics: dict) -> tuple[float, str, dict]:
    score = min(100.0, round(score, 1))
    if score >= 70:
        risk_level = "high"
    elif score >= 40:
        risk_level = "medium"
    else:
        risk_level = "low"
    return score, risk_level, {"reasons": reasons, "metrics_summary": metrics}

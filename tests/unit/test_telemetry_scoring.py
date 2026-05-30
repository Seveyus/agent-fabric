from core.services.telemetry_service import score_github_metrics, score_jira_metrics


def test_score_github_metrics_high_risk():
    score, risk_level, evidence = score_github_metrics(
        {
            "repo_name": "acme/project",
            "default_branch": "main",
            "open_pull_requests": 9,
            "open_issues": 14,
            "commits_last_7d": 0,
            "active_contributors_30d": 1,
            "last_commit_age_days": 10,
            "stale_pull_requests": 3,
        }
    )

    assert score >= 70
    assert risk_level == "high"
    assert evidence["reasons"]


def test_score_jira_metrics_medium_or_high_risk():
    score, risk_level, evidence = score_jira_metrics(
        {
            "project_key": "OPS",
            "open_tickets": 35,
            "blocked_tickets": 2,
            "unassigned_tickets": 6,
            "stale_tickets": 7,
            "overdue_tickets": 3,
        }
    )

    assert score >= 40
    assert risk_level in {"medium", "high"}
    assert any("tickets" in reason for reason in evidence["reasons"])

from core.services.simulation_service import apply_adjustments, build_recommendations, rescore_metrics


def test_apply_adjustments_updates_numeric_fields():
    metrics = {"stale_prs": 4, "active_contributors_30d": 1}
    adjusted = apply_adjustments(metrics, {"stale_prs": -2, "active_contributors_30d": 2})

    assert adjusted["stale_prs"] == 2
    assert adjusted["active_contributors_30d"] == 3


def test_rescore_metrics_github_reduces_when_activity_improves():
    baseline_score, _, _ = rescore_metrics(
        "github",
        {
            "open_pull_requests": 10,
            "commit_velocity_7d": 0,
            "active_contributors_30d": 1,
            "last_commit_age_days": 12,
            "stale_prs": 4,
        },
    )
    improved_score, _, _ = rescore_metrics(
        "github",
        {
            "open_pull_requests": 5,
            "commit_velocity_7d": 7,
            "active_contributors_30d": 3,
            "last_commit_age_days": 1,
            "stale_prs": 1,
        },
    )

    assert improved_score < baseline_score


def test_build_recommendations_mentions_high_leverage_actions():
    recommendations = build_recommendations(
        "jira",
        {"blocked_tickets": 5, "unassigned_tickets": 6, "stale_tickets": 9},
        {"blocked_tickets": 1, "unassigned_tickets": 2, "stale_tickets": 4},
    )

    assert recommendations
    assert any("Unblocking tickets" in item for item in recommendations)

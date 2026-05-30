from datetime import datetime, timedelta, timezone

import httpx


class GitHubTelemetryClient:
    def __init__(self, base_url: str, auth_token: str):
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token

    def collect_repo_metrics(self, repo_full_name: str) -> dict:
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        commits_30d_since = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

        with httpx.Client(timeout=30.0, headers=headers) as client:
            repo = client.get(f"{self.base_url}/repos/{repo_full_name}")
            repo.raise_for_status()

            pulls = client.get(f"{self.base_url}/repos/{repo_full_name}/pulls", params={"state": "open", "per_page": 100})
            pulls.raise_for_status()

            issues = client.get(
                f"{self.base_url}/repos/{repo_full_name}/issues",
                params={"state": "open", "per_page": 100},
            )
            issues.raise_for_status()

            recent_commits = client.get(
                f"{self.base_url}/repos/{repo_full_name}/commits",
                params={"since": since, "per_page": 100},
            )
            recent_commits.raise_for_status()

            commits_30d = client.get(
                f"{self.base_url}/repos/{repo_full_name}/commits",
                params={"since": commits_30d_since, "per_page": 100},
            )
            commits_30d.raise_for_status()

            latest_commit = client.get(
                f"{self.base_url}/repos/{repo_full_name}/commits",
                params={"per_page": 1},
            )
            latest_commit.raise_for_status()

        repo_payload = repo.json()
        pull_items = pulls.json()
        issue_items = [item for item in issues.json() if "pull_request" not in item]
        commit_items = recent_commits.json()
        commit_30d_items = commits_30d.json()
        latest_commit_items = latest_commit.json()

        last_commit_age_days = None
        if latest_commit_items:
            commit_date = latest_commit_items[0]["commit"]["author"]["date"]
            parsed = datetime.fromisoformat(commit_date.replace("Z", "+00:00"))
            last_commit_age_days = (datetime.now(timezone.utc) - parsed).days

        active_contributors = {
            item["author"]["login"]
            for item in commit_30d_items
            if item.get("author") and item["author"].get("login")
        }
        stale_pull_requests = 0
        for item in pull_items:
            updated_at = datetime.fromisoformat(item["updated_at"].replace("Z", "+00:00"))
            if datetime.now(timezone.utc) - updated_at > timedelta(days=7):
                stale_pull_requests += 1

        return {
            "repo_name": repo_payload["full_name"],
            "default_branch": repo_payload["default_branch"],
            "open_pull_requests": len(pull_items),
            "open_issues": len(issue_items),
            "commits_last_7d": len(commit_items),
            "active_contributors_30d": len(active_contributors),
            "last_commit_age_days": last_commit_age_days if last_commit_age_days is not None else 999,
            "stale_pull_requests": stale_pull_requests,
        }

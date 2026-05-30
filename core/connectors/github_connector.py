from datetime import datetime, timezone

import httpx

from core.connectors.base import BaseConnector


class GitHubConnector(BaseConnector):
    provider = "github"

    def authenticate(self) -> dict:
        return {
            "Authorization": f"Bearer {self.integration.auth_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def fetch_raw(self, project_ref: str) -> dict:
        headers = self.authenticate()
        with httpx.Client(timeout=30.0, headers=headers) as client:
            repo = client.get(f"{self.integration.base_url}/repos/{project_ref}")
            repo.raise_for_status()
            pulls = client.get(f"{self.integration.base_url}/repos/{project_ref}/pulls", params={"state": "all", "per_page": 50})
            pulls.raise_for_status()
            issues = client.get(f"{self.integration.base_url}/repos/{project_ref}/issues", params={"state": "all", "per_page": 50})
            issues.raise_for_status()
            commits = client.get(f"{self.integration.base_url}/repos/{project_ref}/commits", params={"per_page": 50})
            commits.raise_for_status()
            contributors = client.get(f"{self.integration.base_url}/repos/{project_ref}/contributors", params={"per_page": 30})
            contributors.raise_for_status()
            releases = client.get(f"{self.integration.base_url}/repos/{project_ref}/releases", params={"per_page": 20})
            if releases.status_code >= 400 and releases.status_code != 404:
                releases.raise_for_status()
        return {
            "repository": repo.json(),
            "pull_requests": pulls.json(),
            "issues": issues.json(),
            "commits": commits.json(),
            "contributors": contributors.json(),
            "releases": [] if releases.status_code == 404 else releases.json(),
        }

    def normalize_entities(self, raw_payload: dict, project_ref: str) -> list[dict]:
        repo = raw_payload["repository"]
        entities = [
            self._entity("Project", f"project:{project_ref}", repo["name"], project_ref, {"provider": "github"}),
            self._entity("Repository", f"repo:{project_ref}", repo["full_name"], project_ref, {"default_branch": repo.get("default_branch")}),
        ]
        for contributor in raw_payload["contributors"]:
            entities.append(
                self._entity("Person", f"github:user:{contributor['login']}", contributor["login"], project_ref, {"contributions": contributor.get("contributions", 0)})
            )
        for pr in raw_payload["pull_requests"]:
            entities.append(
                self._entity("PullRequest", f"github:pr:{project_ref}:{pr['number']}", pr["title"], project_ref, {"state": pr["state"], "number": pr["number"]})
            )
        for issue in raw_payload["issues"]:
            if "pull_request" in issue:
                continue
            entities.append(
                self._entity("Ticket", f"github:issue:{project_ref}:{issue['number']}", issue["title"], project_ref, {"state": issue["state"], "number": issue["number"]})
            )
        for commit in raw_payload["commits"]:
            sha = commit["sha"]
            entities.append(
                self._entity("Commit", f"github:commit:{project_ref}:{sha}", sha[:12], project_ref, {"sha": sha, "message": commit["commit"]["message"][:200]})
            )
        return entities

    def normalize_relations(self, raw_payload: dict, project_ref: str) -> list[dict]:
        relations = [
            self._relation("implements", f"repo:{project_ref}", f"project:{project_ref}", project_ref),
            self._relation("belongs_to", f"project:{project_ref}", f"workspace:github:{self.integration.name}", project_ref),
        ]
        for contributor in raw_payload["contributors"]:
            relations.append(self._relation("works_on", f"github:user:{contributor['login']}", f"project:{project_ref}", project_ref))
        for pr in raw_payload["pull_requests"]:
            pr_ref = f"github:pr:{project_ref}:{pr['number']}"
            relations.append(self._relation("belongs_to", pr_ref, f"repo:{project_ref}", project_ref))
            user = pr.get("user")
            if user:
                relations.append(self._relation("created_by", pr_ref, f"github:user:{user['login']}", project_ref))
        for issue in raw_payload["issues"]:
            if "pull_request" in issue:
                continue
            issue_ref = f"github:issue:{project_ref}:{issue['number']}"
            relations.append(self._relation("affects", issue_ref, f"project:{project_ref}", project_ref))
            assignee = issue.get("assignee")
            if assignee:
                relations.append(self._relation("assigned_to", issue_ref, f"github:user:{assignee['login']}", project_ref))
        for commit in raw_payload["commits"]:
            author = commit.get("author")
            commit_ref = f"github:commit:{project_ref}:{commit['sha']}"
            relations.append(self._relation("belongs_to", commit_ref, f"repo:{project_ref}", project_ref))
            if author:
                relations.append(self._relation("created_by", commit_ref, f"github:user:{author['login']}", project_ref))
        return relations

    def create_snapshots(self, raw_payload: dict, project_ref: str) -> list[dict]:
        issues = [item for item in raw_payload["issues"] if "pull_request" not in item]
        closed_issues = [item for item in issues if item.get("state") == "closed"]
        open_issues = [item for item in issues if item.get("state") == "open"]
        prs = raw_payload["pull_requests"]
        open_prs = [item for item in prs if item.get("state") == "open"]
        stale_prs = [item for item in open_prs if self._age_days(item["updated_at"]) > 7]
        commits = raw_payload["commits"]
        releases = raw_payload["releases"]

        review_delays = []
        for pr in prs:
            created = pr.get("created_at")
            updated = pr.get("updated_at")
            if created and updated:
                review_delays.append(max(0.0, self._hours_between(created, updated)))
        issue_ages = [self._age_days(item["created_at"]) for item in open_issues if item.get("created_at")]
        release_recency = self._age_days(releases[0]["published_at"]) if releases else 365.0

        return [
            self._metric(project_ref, "github", "commit_velocity_7d", float(min(len(commits), 7)), "count"),
            self._metric(project_ref, "github", "stale_prs", float(len(stale_prs)), "count"),
            self._metric(project_ref, "github", "open_issues", float(len(open_issues)), "count"),
            self._metric(project_ref, "github", "closed_issues", float(len(closed_issues)), "count"),
            self._metric(project_ref, "github", "issue_age_avg_days", float(sum(issue_ages) / len(issue_ages)) if issue_ages else 0.0, "days"),
            self._metric(project_ref, "github", "pr_review_delay_avg_hours", float(sum(review_delays) / len(review_delays)) if review_delays else 0.0, "hours"),
            self._metric(project_ref, "github", "release_recency_days", float(release_recency), "days"),
        ]

    def _entity(self, entity_type: str, entity_ref: str, name: str, project_ref: str, attributes: dict) -> dict:
        return {"entity_type": entity_type, "entity_ref": entity_ref, "name": name, "project_ref": project_ref, "attributes": attributes}

    def _relation(self, relation_type: str, source_ref: str, target_ref: str, project_ref: str, attributes: dict | None = None) -> dict:
        return {"relation_type": relation_type, "source_ref": source_ref, "target_ref": target_ref, "project_ref": project_ref, "attributes": attributes or {}}

    def _metric(self, project_ref: str, source: str, name: str, value: float, unit: str) -> dict:
        return {"project_ref": project_ref, "metric_source": source, "metric_name": name, "metric_value": value, "metric_unit": unit, "dimensions": {}}

    def _age_days(self, iso_date: str) -> float:
        parsed = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
        return float((datetime.now(timezone.utc) - parsed).days)

    def _hours_between(self, start_iso: str, end_iso: str) -> float:
        start = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
        return (end - start).total_seconds() / 3600.0

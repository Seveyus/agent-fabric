from datetime import datetime, timezone

import httpx

from core.connectors.base import BaseConnector


class JiraConnector(BaseConnector):
    provider = "jira"

    def authenticate(self) -> dict:
        return {"Authorization": self.authorization_header("Bearer"), "Accept": "application/json"}

    def discover_recent_projects(self, limit: int = 10) -> list[dict]:
        headers = self.authenticate()
        with httpx.Client(timeout=30.0, headers=headers) as client:
            recent_issues = client.get(
                f"{self.integration.base_url}/rest/api/3/search",
                params={
                    "jql": "ORDER BY updated DESC",
                    "maxResults": min(max(limit * 8, 20), 100),
                    "fields": "project,updated",
                },
            )
            recent_issues.raise_for_status()

            projects = []
            seen_keys = set()
            for issue in recent_issues.json().get("issues", []):
                fields = issue.get("fields", {})
                project = fields.get("project") or {}
                project_key = project.get("key")
                if not project_key or project_key in seen_keys:
                    continue
                seen_keys.add(project_key)
                projects.append(
                    {
                        "project_ref": project_key,
                        "display_name": project.get("name", project_key),
                        "provider": self.provider,
                        "last_activity_at": fields.get("updated"),
                        "metadata": {
                            "project_id": project.get("id"),
                            "project_type": project.get("projectTypeKey"),
                        },
                    }
                )
                if len(projects) >= limit:
                    return projects

            response = client.get(
                f"{self.integration.base_url}/rest/api/3/project/search",
                params={"maxResults": min(max(limit, 1), 50)},
            )
            response.raise_for_status()

        fallback_projects = projects[:]
        for project in response.json().get("values", []):
            project_key = project.get("key")
            if not project_key or project_key in seen_keys:
                continue
            seen_keys.add(project_key)
            fallback_projects.append(
                {
                    "project_ref": project_key,
                    "display_name": project.get("name", project_key),
                    "provider": self.provider,
                    "last_activity_at": None,
                    "metadata": {
                        "project_id": project.get("id"),
                        "project_type": project.get("projectTypeKey"),
                    },
                }
            )
            if len(fallback_projects) >= limit:
                break
        return fallback_projects[:limit]

    def fetch_raw(self, project_ref: str) -> dict:
        headers = self.authenticate()
        params = {
            "jql": f"project={project_ref} ORDER BY updated DESC",
            "maxResults": 100,
            "fields": "summary,status,assignee,priority,updated,created,duedate,parent,issuetype",
        }
        with httpx.Client(timeout=30.0, headers=headers) as client:
            project = client.get(f"{self.integration.base_url}/rest/api/3/project/{project_ref}")
            project.raise_for_status()
            search = client.get(f"{self.integration.base_url}/rest/api/3/search", params=params)
            search.raise_for_status()
        return {"project": project.json(), "issues": search.json().get("issues", [])}

    def normalize_entities(self, raw_payload: dict, project_ref: str) -> list[dict]:
        project = raw_payload["project"]
        entities = [
            self._entity("Project", f"project:{project_ref}", project.get("name", project_ref), project_ref, {"provider": "jira"}),
        ]
        seen_people = set()
        for issue in raw_payload["issues"]:
            fields = issue["fields"]
            issue_type = fields.get("issuetype", {}).get("name", "Task")
            mapped_type = "Risk" if "risk" in issue_type.lower() else "Task" if issue_type.lower() in {"task", "story"} else "Ticket"
            entities.append(
                self._entity(mapped_type, f"jira:issue:{project_ref}:{issue['key']}", fields.get("summary", issue["key"]), project_ref, {
                    "status": fields.get("status", {}).get("name"),
                    "priority": fields.get("priority", {}).get("name"),
                    "issue_type": issue_type,
                })
            )
            assignee = fields.get("assignee")
            if assignee and assignee.get("accountId") not in seen_people:
                seen_people.add(assignee["accountId"])
                entities.append(
                    self._entity("Person", f"jira:user:{assignee['accountId']}", assignee.get("displayName", assignee["accountId"]), project_ref, {})
                )
        return entities

    def normalize_relations(self, raw_payload: dict, project_ref: str) -> list[dict]:
        relations = [self._relation("belongs_to", f"project:{project_ref}", f"workspace:jira:{self.integration.name}", project_ref)]
        for issue in raw_payload["issues"]:
            fields = issue["fields"]
            issue_ref = f"jira:issue:{project_ref}:{issue['key']}"
            relations.append(self._relation("affects", issue_ref, f"project:{project_ref}", project_ref))
            assignee = fields.get("assignee")
            if assignee:
                relations.append(self._relation("assigned_to", issue_ref, f"jira:user:{assignee['accountId']}", project_ref))
            parent = fields.get("parent")
            if parent:
                relations.append(self._relation("depends_on", issue_ref, f"jira:issue:{project_ref}:{parent['key']}", project_ref))
            status_name = (fields.get("status", {}) or {}).get("name", "").lower()
            if "block" in status_name:
                relations.append(self._relation("blocks", issue_ref, f"project:{project_ref}", project_ref))
        return relations

    def create_snapshots(self, raw_payload: dict, project_ref: str) -> list[dict]:
        issues = raw_payload["issues"]
        blocked = 0
        overdue = 0
        unassigned = 0
        age_days = []
        closed_like = 0
        for issue in issues:
            fields = issue["fields"]
            status_name = (fields.get("status", {}) or {}).get("name", "").lower()
            if "block" in status_name:
                blocked += 1
            if not fields.get("assignee"):
                unassigned += 1
            if fields.get("duedate"):
                due_dt = datetime.fromisoformat(f"{fields['duedate']}T00:00:00+00:00")
                if due_dt < datetime.now(timezone.utc):
                    overdue += 1
            if fields.get("created"):
                age_days.append(self._age_days(fields["created"]))
            if status_name in {"done", "closed", "resolved"}:
                closed_like += 1
        cycle_time_avg = float(sum(age_days) / len(age_days)) if age_days else 0.0
        velocity_proxy = float(closed_like)
        return [
            self._metric(project_ref, "jira", "backlog_size", float(len(issues)), "count"),
            self._metric(project_ref, "jira", "blocked_tickets", float(blocked), "count"),
            self._metric(project_ref, "jira", "overdue_tickets", float(overdue), "count"),
            self._metric(project_ref, "jira", "unassigned_tickets", float(unassigned), "count"),
            self._metric(project_ref, "jira", "cycle_time_avg_days", cycle_time_avg, "days"),
            self._metric(project_ref, "jira", "velocity_proxy", velocity_proxy, "count"),
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

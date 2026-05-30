from datetime import datetime, timezone
import httpx


class JiraTelemetryClient:
    def __init__(self, base_url: str, auth_token: str):
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token

    def collect_project_metrics(self, project_key: str) -> dict:
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Accept": "application/json",
        }
        jql = f"project={project_key} ORDER BY updated DESC"
        url = f"{self.base_url}/rest/api/3/search/jql"
        params = {
            "jql": jql,
            "maxResults": 100,
            "fields": "status,assignee,updated,duedate",
        }

        with httpx.Client(timeout=30.0, headers=headers) as client:
            response = client.get(url, params=params)
            if response.status_code == 404:
                fallback_url = f"{self.base_url}/rest/api/3/search"
                response = client.get(fallback_url, params=params)
            response.raise_for_status()

        payload = response.json()
        issues = payload.get("issues", [])

        blocked = 0
        unassigned = 0
        stale = 0
        overdue = 0

        now = datetime.now(timezone.utc)
        for issue in issues:
            fields = issue.get("fields", {})
            status_name = (fields.get("status") or {}).get("name", "").lower()
            updated = fields.get("updated")
            due_date = fields.get("duedate")
            assignee = fields.get("assignee")

            if "block" in status_name:
                blocked += 1
            if not assignee:
                unassigned += 1
            if updated:
                updated_dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                if (now - updated_dt).days > 14:
                    stale += 1
            if due_date:
                due_dt = datetime.fromisoformat(f"{due_date}T00:00:00+00:00")
                if due_dt < now:
                    overdue += 1

        return {
            "project_key": project_key,
            "open_tickets": len(issues),
            "blocked_tickets": blocked,
            "unassigned_tickets": unassigned,
            "stale_tickets": stale,
            "overdue_tickets": overdue,
        }

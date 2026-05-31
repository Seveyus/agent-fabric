import httpx

from core.connectors.base import BaseConnector


class NotionConnector(BaseConnector):
    provider = "notion"

    def authenticate(self) -> dict:
        return {
            "Authorization": self.authorization_header("Bearer"),
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
        }

    def discover_recent_projects(self, limit: int = 10) -> list[dict]:
        headers = self.authenticate()
        with httpx.Client(timeout=30.0, headers=headers) as client:
            response = client.post(
                f"{self.integration.base_url}/v1/search",
                json={"page_size": min(max(limit * 3, 10), 50)},
            )
            response.raise_for_status()

        projects = []
        seen_refs = set()
        for item in response.json().get("results", []):
            title = self._extract_title(item).strip()
            if not title:
                continue
            project_ref = self._project_ref(item, title)
            if project_ref in seen_refs:
                continue
            seen_refs.add(project_ref)
            projects.append(
                {
                    "project_ref": project_ref,
                    "display_name": title,
                    "provider": self.provider,
                    "last_activity_at": item.get("last_edited_time"),
                    "metadata": {
                        "object_type": item.get("object"),
                        "notion_id": item.get("id"),
                    },
                }
            )
            if len(projects) >= limit:
                break
        return projects

    def fetch_raw(self, project_ref: str) -> dict:
        headers = self.authenticate()
        with httpx.Client(timeout=30.0, headers=headers) as client:
            response = client.post(f"{self.integration.base_url}/v1/search", json={"page_size": 50})
            response.raise_for_status()
        return {"results": response.json().get("results", []), "project_ref": project_ref}

    def normalize_entities(self, raw_payload: dict, project_ref: str) -> list[dict]:
        entities = [self._entity("Project", f"project:{project_ref}", project_ref, project_ref, {"provider": "notion"})]
        for item in raw_payload["results"]:
            object_type = item.get("object")
            parent_type = item.get("parent", {}).get("type")
            title = self._extract_title(item)
            entity_type = "Document"
            if object_type == "database":
                entity_type = "Task"
            if "decision" in title.lower():
                entity_type = "Decision"
            entities.append(
                self._entity(entity_type, f"notion:{object_type}:{item['id']}", title or item["id"], project_ref, {"parent_type": parent_type})
            )
        return entities

    def normalize_relations(self, raw_payload: dict, project_ref: str) -> list[dict]:
        relations = [self._relation("belongs_to", f"project:{project_ref}", f"workspace:notion:{self.integration.name}", project_ref)]
        for item in raw_payload["results"]:
            title = self._extract_title(item)
            entity_ref = f"notion:{item.get('object')}:{item['id']}"
            relations.append(self._relation("mentions", entity_ref, f"project:{project_ref}", project_ref))
            if "decision" in title.lower():
                relations.append(self._relation("related_to", entity_ref, f"project:{project_ref}", project_ref))
        return relations

    def create_snapshots(self, raw_payload: dict, project_ref: str) -> list[dict]:
        results = raw_payload["results"]
        decision_count = 0
        stale_docs = 0
        project_mentions = 0
        for item in results:
            title = self._extract_title(item)
            if "decision" in title.lower():
                decision_count += 1
            if "project" in title.lower():
                project_mentions += 1
            if item.get("last_edited_time", "").startswith("2024") or item.get("last_edited_time", "").startswith("2025-01"):
                stale_docs += 1
        undocumented = 0.0 if project_mentions else 1.0
        return [
            self._metric(project_ref, "notion", "recent_decisions", float(decision_count), "count"),
            self._metric(project_ref, "notion", "undocumented_projects", undocumented, "flag"),
            self._metric(project_ref, "notion", "decisions_linked_to_projects", float(project_mentions), "count"),
            self._metric(project_ref, "notion", "stale_docs", float(stale_docs), "count"),
        ]

    def _extract_title(self, item: dict) -> str:
        if item.get("object") == "page":
            props = item.get("properties", {})
            for value in props.values():
                if value.get("type") == "title":
                    parts = [part.get("plain_text", "") for part in value.get("title", [])]
                    return "".join(parts)
        if item.get("object") == "database":
            return "".join(part.get("plain_text", "") for part in item.get("title", []))
        return item.get("id", "")

    def _project_ref(self, item: dict, title: str) -> str:
        slug = "-".join(part for part in "".join(ch.lower() if ch.isalnum() else " " for ch in title).split() if part)
        return slug or f"notion-{item['id'][:8]}"

    def _entity(self, entity_type: str, entity_ref: str, name: str, project_ref: str, attributes: dict) -> dict:
        return {"entity_type": entity_type, "entity_ref": entity_ref, "name": name, "project_ref": project_ref, "attributes": attributes}

    def _relation(self, relation_type: str, source_ref: str, target_ref: str, project_ref: str, attributes: dict | None = None) -> dict:
        return {"relation_type": relation_type, "source_ref": source_ref, "target_ref": target_ref, "project_ref": project_ref, "attributes": attributes or {}}

    def _metric(self, project_ref: str, source: str, name: str, value: float, unit: str) -> dict:
        return {"project_ref": project_ref, "metric_source": source, "metric_name": name, "metric_value": value, "metric_unit": unit, "dimensions": {}}

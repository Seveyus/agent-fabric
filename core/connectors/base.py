from abc import ABC, abstractmethod


class BaseConnector(ABC):
    provider: str

    def __init__(self, integration):
        self.integration = integration

    def authorization_header(self, default_scheme: str = "Bearer") -> str:
        raw_value = (self.integration.auth_token or "").strip()
        lowered = raw_value.lower()
        if lowered.startswith("bearer ") or lowered.startswith("token ") or lowered.startswith("basic "):
            return raw_value
        return f"{default_scheme} {raw_value}"

    @abstractmethod
    def authenticate(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def discover_recent_projects(self, limit: int = 10) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def fetch_raw(self, project_ref: str) -> dict:
        raise NotImplementedError

    @abstractmethod
    def normalize_entities(self, raw_payload: dict, project_ref: str) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def normalize_relations(self, raw_payload: dict, project_ref: str) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def create_snapshots(self, raw_payload: dict, project_ref: str) -> list[dict]:
        raise NotImplementedError

    def sync(self, project_ref: str) -> dict:
        self.authenticate()
        raw_payload = self.fetch_raw(project_ref)
        entities = self.normalize_entities(raw_payload, project_ref)
        relations = self.normalize_relations(raw_payload, project_ref)
        snapshots = self.create_snapshots(raw_payload, project_ref)
        return {
            "raw_payload": raw_payload,
            "entities": entities,
            "relations": relations,
            "metric_snapshots": snapshots,
        }

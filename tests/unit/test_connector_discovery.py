import core.services.connector_sync_service as connector_sync_service


class _FakeConnector:
    def __init__(self, integration):
        self.integration = integration

    def discover_recent_projects(self, limit=10):
        return [
            {
                "project_ref": "acme/api",
                "display_name": "acme/api",
                "provider": "github",
                "last_activity_at": None,
                "metadata": {},
            },
            {
                "project_ref": "acme/api",
                "display_name": "Duplicate",
                "provider": "github",
                "last_activity_at": None,
                "metadata": {},
            },
            {
                "project_ref": "acme/web",
                "display_name": "acme/web",
                "provider": "github",
                "last_activity_at": None,
                "metadata": {},
            },
        ][:limit]


class _FakeRun:
    def __init__(self, project_ref, status="completed"):
        self.id = f"sync-{project_ref.replace('/', '-')}"
        self.integration_id = "int_test"
        self.provider = "github"
        self.project_ref = project_ref
        self.status = status
        self.raw_count = 1
        self.entity_count = 1
        self.relation_count = 1
        self.snapshot_count = 1
        self.error_message = None
        self.started_at = None
        self.completed_at = None


def test_discover_recent_projects_dedupes_results(monkeypatch):
    integration = type("Integration", (), {"provider": "github", "id": "int_test"})()
    monkeypatch.setitem(connector_sync_service.CONNECTOR_MAP, "github", _FakeConnector)

    projects = connector_sync_service.discover_recent_projects(integration, limit=5)

    assert [item["project_ref"] for item in projects] == ["acme/api", "acme/web"]


def test_sync_recent_projects_returns_batch_summary(monkeypatch):
    integration = type("Integration", (), {"provider": "github", "id": "int_test"})()
    monkeypatch.setitem(connector_sync_service.CONNECTOR_MAP, "github", _FakeConnector)
    monkeypatch.setattr(
        connector_sync_service,
        "run_connector_sync",
        lambda db, integration, project_ref: _FakeRun(project_ref),
    )

    result = connector_sync_service.sync_recent_projects(db=None, integration=integration, limit=5)

    assert result["discovered_count"] == 2
    assert result["synced_count"] == 2
    assert result["failed_count"] == 0
    assert [item.project_ref for item in result["sync_runs"]] == ["acme/api", "acme/web"]

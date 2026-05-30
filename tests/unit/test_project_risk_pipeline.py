from core.agents.extractor_agent import ExtractorAgent
from core.agents.risk_analyst_agent import RiskAnalystAgent
from core.orchestrator.pipeline_runner import build_project_snapshot


def test_extractor_captures_project_risk_signals():
    parsed_docs = [
        {
            "document_id": "doc_1",
            "text": (
                "Action: finish release checklist\n"
                "Owner: Alice Martin\n"
                "Blocked: waiting on infra sign-off\n"
                "Dependency: pending mobile validation\n"
                "Risk: launch may slip\n"
            ),
        }
    ]

    extracted = ExtractorAgent().run(parsed_docs)

    assert extracted[0]["owners"] == ["Alice Martin"]
    assert len(extracted[0]["action_lines"]) == 1
    assert len(extracted[0]["blocker_lines"]) == 1
    assert len(extracted[0]["dependency_lines"]) == 1
    assert len(extracted[0]["risk_lines"]) == 1


def test_risk_analyst_emits_project_risk_score():
    extracted = [
        {
            "owners": [],
            "action_lines": ["Action: ship release", "Action: validate QA"],
            "dates": [],
            "risk_lines": ["Risk: schedule may slip"],
            "blocker_lines": ["Blocked: waiting on API access"],
            "dependency_lines": ["Dependency: vendor approval"],
        }
    ]

    findings = RiskAnalystAgent().run(extracted, retrieved_chunks=[object()], objective="Ship release safely")
    score_finding = next(item for item in findings if item["kind"] == "project_risk_score")

    assert score_finding["score"] >= 40
    assert score_finding["risk_level"] in {"medium", "high"}
    assert any(item["kind"] == "ownership_gap" for item in findings)
    assert any(item["kind"] == "delivery_friction" for item in findings)


def test_build_project_snapshot_uses_score_finding():
    snapshot = build_project_snapshot(
        documents=[object(), object()],
        chunks=[object()],
        extracted=[
            {
                "owners": ["Alice"],
                "action_lines": ["Action: ship release", "Action: validate QA"],
                "dates": ["2026-06-01"],
                "risk_lines": ["Risk: schedule may slip"],
                "blocker_lines": [],
                "dependency_lines": ["Dependency: vendor approval"],
            }
        ],
        findings=[
            {
                "kind": "project_risk_score",
                "score": 58.0,
                "risk_level": "medium",
            }
        ],
    )

    assert snapshot["total_documents"] == 2
    assert snapshot["total_chunks"] == 1
    assert snapshot["actions_detected"] == 2
    assert snapshot["owner_coverage_ratio"] == 0.5
    assert snapshot["risk_score"] == 58.0
    assert snapshot["risk_level"] == "medium"

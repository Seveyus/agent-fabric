from datetime import date

from sqlalchemy.orm import Session

from apps.api.config import settings
from core.agents.chunker_embed_agent import ChunkerEmbedAgent
from core.agents.extractor_agent import ExtractorAgent
from core.agents.parser_agent import ParserAgent
from core.agents.report_agent import ReportAgent
from core.agents.retriever_agent import RetrieverAgent
from core.agents.risk_analyst_agent import RiskAnalystAgent
from core.evidence.linker import link_evidence
from core.orchestrator.step_executor import execute_step
from core.services.artifact_service import write_report_artifact
from core.services.finding_service import persist_findings
from core.services.knowledge_graph_service import ingest_document_snapshot_into_graph
from core.services.project_snapshot_service import persist_project_snapshot
from core.services.run_service import mark_run_step
from db.models.document import Document


class ProjectRiskPipelineRunner:
    def __init__(self):
        self.parser = ParserAgent()
        self.extractor = ExtractorAgent()
        self.chunker = ChunkerEmbedAgent()
        self.retriever = RetrieverAgent()
        self.analyst = RiskAnalystAgent()
        self.reporter = ReportAgent()

    def run(self, db: Session, run, job):
        mark_run_step(db, run, "load_documents")
        documents = db.query(Document).filter(Document.id.in_(job.input_document_ids)).all()

        extracted_root = f"{settings.storage_root}/extracted"

        mark_run_step(db, run, "parse_documents")
        parsed_docs = execute_step(db, run.id, "parse_documents", self.parser.run, documents, extracted_root)

        mark_run_step(db, run, "extract_signals")
        extracted = execute_step(db, run.id, "extract_signals", self.extractor.run, parsed_docs)

        mark_run_step(db, run, "chunk_and_embed")
        chunks = execute_step(db, run.id, "chunk_and_embed", self.chunker.run, parsed_docs, db)

        mark_run_step(db, run, "retrieve_evidence")
        retrieved_chunks = execute_step(
            db,
            run.id,
            "retrieve_evidence",
            self.retriever.run,
            job.objective,
            db,
        )

        documents_by_id = {d.id: d for d in documents}
        linked_evidence = link_evidence(retrieved_chunks, documents_by_id, max_items=2)

        mark_run_step(db, run, "score_project_risk")
        findings = execute_step(
            db,
            run.id,
            "score_project_risk",
            self.analyst.run,
            extracted,
            retrieved_chunks,
            job.objective,
        )

        for finding in findings:
            finding["evidence"] = linked_evidence

        mark_run_step(db, run, "persist_snapshot")
        snapshot = build_project_snapshot(job, documents, chunks, extracted, findings)
        persist_project_snapshot(db, run.id, snapshot)
        ingest_document_snapshot_into_graph(db, run.id, snapshot, findings)

        mark_run_step(db, run, "persist_findings")
        persist_findings(db, run.id, findings)

        mark_run_step(db, run, "generate_report")
        markdown = execute_step(db, run.id, "generate_report", self.reporter.run, job.objective, findings)
        write_report_artifact(db, run.id, markdown)


def build_project_snapshot(job, documents: list[Document], chunks: list, extracted: list[dict], findings: list[dict]) -> dict:
    score_finding = next((item for item in findings if item["kind"] == "project_risk_score"), None)
    owner_count = sum(len(item["owners"]) for item in extracted)
    action_count = sum(len(item["action_lines"]) for item in extracted)
    date_count = sum(len(item["dates"]) for item in extracted)
    risk_count = sum(len(item.get("risk_lines", [])) for item in extracted)
    blocker_count = sum(len(item.get("blocker_lines", [])) for item in extracted)
    dependency_count = sum(len(item.get("dependency_lines", [])) for item in extracted)
    owner_coverage_ratio = owner_count / action_count if action_count else 1.0

    return {
        "project_ref": getattr(job, "project_ref", None),
        "snapshot_date": date.today(),
        "total_documents": len(documents),
        "total_chunks": len(chunks),
        "owners_detected": owner_count,
        "actions_detected": action_count,
        "dates_detected": date_count,
        "risks_detected": risk_count,
        "blockers_detected": blocker_count,
        "dependencies_detected": dependency_count,
        "owner_coverage_ratio": round(owner_coverage_ratio, 3),
        "risk_score": score_finding["score"] if score_finding else 0.0,
        "risk_level": score_finding["risk_level"] if score_finding else "low",
    }

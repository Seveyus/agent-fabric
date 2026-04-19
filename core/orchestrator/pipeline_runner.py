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
from core.services.run_service import mark_run_step
from db.models.document import Document


class ProjectAuditPipelineRunner:
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

        mark_run_step(db, run, "analyze_risks")
        findings = execute_step(
            db,
            run.id,
            "analyze_risks",
            self.analyst.run,
            extracted,
            retrieved_chunks,
            job.objective,
        )

        for finding in findings:
            finding["evidence"] = linked_evidence

        mark_run_step(db, run, "persist_findings")
        persist_findings(db, run.id, findings)

        mark_run_step(db, run, "generate_report")
        markdown = execute_step(db, run.id, "generate_report", self.reporter.run, job.objective, findings)
        write_report_artifact(db, run.id, markdown)

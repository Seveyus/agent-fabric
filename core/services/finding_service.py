from uuid import uuid4

from db.models.evidence_ref import EvidenceRef
from db.models.finding import Finding


def persist_findings(db, run_id: str, findings: list[dict]) -> list[Finding]:
    finding_rows = []
    for item in findings:
        finding = Finding(
            id=f"finding_{uuid4().hex[:12]}",
            run_id=run_id,
            kind=item["kind"],
            severity=item["severity"],
            title=item["title"],
            summary=item["summary"],
            confidence=item["confidence"],
        )
        db.add(finding)
        db.flush()
        for ev in item.get("evidence", []):
            db.add(
                EvidenceRef(
                    id=f"ev_{uuid4().hex[:12]}",
                    finding_id=finding.id,
                    document_id=ev["document_id"],
                    chunk_id=ev.get("chunk_id"),
                    quote=ev["quote"],
                    source_file=ev["source_file"],
                    page_number=ev.get("page_number"),
                )
            )
        finding_rows.append(finding)
    db.commit()
    return finding_rows

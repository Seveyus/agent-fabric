from pathlib import Path
from uuid import uuid4

from core.parsing.normalize import parse_document
from db.models.document import Document


class ParserAgent:
    name = "parser"

    def run(self, documents: list[Document], extracted_root: str) -> list[dict]:
        results = []
        Path(extracted_root).mkdir(parents=True, exist_ok=True)
        for doc in documents:
            text = parse_document(doc.storage_path)
            extracted_path = Path(extracted_root) / f"{doc.id}.txt"
            extracted_path.write_text(text, encoding="utf-8")
            doc.status = "parsed"
            results.append({"document_id": doc.id, "text": text, "artifact_id": f"artifact_{uuid4().hex[:12]}"})
        return results

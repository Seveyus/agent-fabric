from pathlib import Path

from core.parsing.docx import parse_docx
from core.parsing.pdf import parse_pdf
from core.parsing.text import parse_text_file
from core.parsing.xlsx import parse_xlsx


def parse_document(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix == ".docx":
        return parse_docx(path)
    if suffix in {".txt", ".md", ".csv"}:
        return parse_text_file(path)
    if suffix == ".xlsx":
        return parse_xlsx(path)
    raise ValueError(f"Unsupported file type: {suffix}")

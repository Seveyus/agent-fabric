from pathlib import Path


def parse_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")

def format_evidence_markdown(evidence: list[dict]) -> str:
    lines = []
    for item in evidence:
        lines.append(
            f"- {item['source_file']} / chunk {item['chunk_id']}: {item['quote'][:180].strip()}"
        )
    return "\n".join(lines)

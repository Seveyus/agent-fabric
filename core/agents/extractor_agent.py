import re


class ExtractorAgent:
    name = "extractor"

    OWNER_PATTERN = re.compile(r"(owner|responsible)[:\s-]+([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?)", re.IGNORECASE)
    DATE_PATTERN = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b")
    ACTION_PATTERN = re.compile(r"\b(action|todo|next step|follow up)\b", re.IGNORECASE)

    def run(self, parsed_docs: list[dict]) -> list[dict]:
        extracted = []
        for item in parsed_docs:
            text = item["text"]
            owners = [m.group(2) for m in self.OWNER_PATTERN.finditer(text)]
            dates = [m.group(1) for m in self.DATE_PATTERN.finditer(text)]
            action_hits = [line.strip() for line in text.splitlines() if self.ACTION_PATTERN.search(line)]
            extracted.append(
                {
                    "document_id": item["document_id"],
                    "owners": owners,
                    "dates": dates,
                    "action_lines": action_hits[:50],
                    "confidence": 0.72,
                }
            )
        return extracted

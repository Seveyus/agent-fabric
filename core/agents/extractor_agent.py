"""
LLM-powered causal extraction agent.

Replaces the previous regex-based extractor with structured LLM extraction
of causal decision triplets from project artifacts (PR descriptions,
issue comments, Notion pages, commit messages).

Each extracted decision becomes a node in the temporal knowledge graph with:
  - The decision text
  - The alternatives that were rejected
  - The rationale (WHY this was chosen)
  - Causal edges: triggered_by, constrained_by, overruled_by, leads_to

Output is designed to feed core/services/knowledge_graph_service.py
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

EXTRACTION_PROMPT = """\
You are analyzing a software project artifact (PR description, issue, commit message, or doc page).

Extract any architectural or technical decisions. For each decision found, return a JSON object.
If no clear decision is present, return {"decision": null}.

Artifact text:
{text}

Return JSON with this schema:
{{
  "decision": "brief description of the decision made, or null if none",
  "alternatives_rejected": ["option A", "option B"],
  "rationale": "why this decision was made (constraint, incident, performance, etc.)",
  "decision_maker": "GitHub login or team name if identifiable, else null",
  "causal_type": "one of: triggered_by | constrained_by | leads_to | overruled_by | general",
  "reversed_by": "reference to PR/issue that reversed this, or null",
  "confidence": 0.0
}}

Rules:
- confidence: 0.9 if the rationale is explicit, 0.6 if inferred, 0.3 if very uncertain
- causal_type: "triggered_by" if the decision was forced by an incident or external event;
               "constrained_by" if limited by a technical/legal/resource constraint;
               "leads_to" if it will causally produce a follow-up;
               "overruled_by" if it reversed a prior decision;
               "general" otherwise
- If the text mentions reverting or rolling back a previous decision, set reversed_by
- Keep decision and rationale under 200 characters each
"""


class ExtractorAgent:
    name = "extractor"

    def __init__(self, llm_client=None) -> None:
        self._llm = llm_client

    def _get_client(self):
        if self._llm:
            return self._llm
        # Lazy import to avoid circular dependency at import time
        from core.llm.ollama_client import OllamaClient
        return OllamaClient()

    def run(self, parsed_docs: list[dict]) -> list[dict]:
        """
        Extract causal decisions from a list of parsed documents.

        Each item in parsed_docs must have:
            document_id  str
            text         str

        Returns list of extraction results, one per document.
        """
        client = self._get_client()
        results = []
        for item in parsed_docs:
            text = (item.get("text") or "").strip()
            if not text:
                results.append(self._empty(item["document_id"]))
                continue

            # Truncate — LLMs don't need more than 1500 chars for decision extraction
            truncated = text[:1500]
            prompt = EXTRACTION_PROMPT.format(text=truncated)

            try:
                raw = client.generate_json(prompt)
            except Exception as exc:
                log.warning("LLM extraction failed for %s: %s", item["document_id"], exc)
                results.append(self._empty(item["document_id"]))
                continue

            results.append(self._parse_result(item["document_id"], raw, text))

        return results

    def _parse_result(self, doc_id: str, raw: dict, original_text: str) -> dict:
        decision = raw.get("decision")
        if not decision or decision == "null":
            return self._empty(doc_id)

        return {
            "document_id": doc_id,
            "decision": str(decision)[:200],
            "alternatives_rejected": raw.get("alternatives_rejected") or [],
            "rationale": str(raw.get("rationale") or "")[:200],
            "decision_maker": raw.get("decision_maker"),
            "causal_type": raw.get("causal_type", "general"),
            "reversed_by": raw.get("reversed_by"),
            "confidence": float(raw.get("confidence", 0.5)),
            "has_decision": True,
            # Legacy fields — kept for compatibility with downstream consumers
            "owners": [raw["decision_maker"]] if raw.get("decision_maker") else [],
            "dates": [],
            "action_lines": [],
            "risk_lines": [],
            "blocker_lines": [],
            "dependency_lines": [],
        }

    def _empty(self, doc_id: str) -> dict:
        return {
            "document_id": doc_id,
            "decision": None,
            "alternatives_rejected": [],
            "rationale": "",
            "decision_maker": None,
            "causal_type": None,
            "reversed_by": None,
            "confidence": 0.0,
            "has_decision": False,
            "owners": [],
            "dates": [],
            "action_lines": [],
            "risk_lines": [],
            "blocker_lines": [],
            "dependency_lines": [],
        }

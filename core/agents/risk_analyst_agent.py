from collections import Counter


class RiskAnalystAgent:
    name = "risk_analyst"

    def run(self, extracted: list[dict], retrieved_chunks: list, objective: str) -> list[dict]:
        findings = []
        owner_count = sum(len(item["owners"]) for item in extracted)
        action_count = sum(len(item["action_lines"]) for item in extracted)
        date_count = sum(len(item["dates"]) for item in extracted)

        if action_count > owner_count:
            findings.append(
                {
                    "kind": "missing_owner",
                    "severity": "high",
                    "title": "Actions détectées sans ownership suffisant",
                    "summary": f"{action_count} lignes d'action ont été trouvées pour seulement {owner_count} owners détectés.",
                    "confidence": 0.83,
                }
            )

        if action_count > 0 and date_count == 0:
            findings.append(
                {
                    "kind": "missing_deadline",
                    "severity": "medium",
                    "title": "Actions sans dates explicites",
                    "summary": "Des actions ont été détectées sans échéance claire dans les documents analysés.",
                    "confidence": 0.75,
                }
            )

        all_dates = [date for item in extracted for date in item["dates"]]
        dupes = [date for date, count in Counter(all_dates).items() if count > 3]
        if dupes:
            findings.append(
                {
                    "kind": "date_hotspot",
                    "severity": "low",
                    "title": "Concentration de dates récurrentes",
                    "summary": f"Certaines dates reviennent fortement ({', '.join(dupes[:3])}), ce qui peut signaler un goulot ou un jalon critique.",
                    "confidence": 0.61,
                }
            )

        if not findings and retrieved_chunks:
            findings.append(
                {
                    "kind": "review_required",
                    "severity": "low",
                    "title": "Revue humaine recommandée",
                    "summary": f"Aucun signal fort automatique pour l'objectif '{objective}', mais des éléments ont été récupérés pour inspection.",
                    "confidence": 0.48,
                }
            )
        return findings

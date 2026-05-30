from core.evidence.formatter import format_evidence_markdown


class ReportAgent:
    name = "reporter"

    def run(self, objective: str, findings: list[dict]) -> str:
        lines = [
            "# Project Risk Report",
            "",
            f"**Objective**: {objective}",
            "",
            "## Findings",
            "",
        ]
        if not findings:
            lines.append("No findings.")
            return "\n".join(lines)

        for idx, finding in enumerate(findings, start=1):
            lines.append(f"### {idx}. {finding['title']}")
            lines.append(f"- Kind: {finding['kind']}")
            lines.append(f"- Severity: {finding['severity']}")
            lines.append(f"- Confidence: {finding['confidence']}")
            if "score" in finding:
                lines.append(f"- Score: {finding['score']}/100")
            if "risk_level" in finding:
                lines.append(f"- Risk level: {finding['risk_level']}")
            lines.append(f"- Summary: {finding['summary']}")
            evidence = finding.get("evidence", [])
            if evidence:
                lines.append("- Evidence:")
                lines.append(format_evidence_markdown(evidence))
            lines.append("")
        return "\n".join(lines)

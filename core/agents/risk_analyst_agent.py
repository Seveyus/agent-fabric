from collections import Counter


class RiskAnalystAgent:
    name = "risk_analyst"

    def run(self, extracted: list[dict], retrieved_chunks: list, objective: str) -> list[dict]:
        findings = []
        owner_count = sum(len(item["owners"]) for item in extracted)
        action_count = sum(len(item["action_lines"]) for item in extracted)
        date_count = sum(len(item["dates"]) for item in extracted)
        risk_count = sum(len(item.get("risk_lines", [])) for item in extracted)
        blocker_count = sum(len(item.get("blocker_lines", [])) for item in extracted)
        dependency_count = sum(len(item.get("dependency_lines", [])) for item in extracted)
        owner_coverage = owner_count / action_count if action_count else 1.0

        risk_score = 5.0
        risk_score += min(30.0, max(0.0, (1.0 - owner_coverage) * 35.0))
        risk_score += min(20.0, blocker_count * 8.0)
        risk_score += min(15.0, dependency_count * 4.0)
        risk_score += min(20.0, risk_count * 3.0)
        if action_count > 0 and date_count == 0:
            risk_score += 10.0
        risk_score = min(100.0, round(risk_score, 1))

        if risk_score >= 70:
            risk_level = "high"
        elif risk_score >= 40:
            risk_level = "medium"
        else:
            risk_level = "low"

        if action_count > owner_count:
            findings.append(
                {
                    "kind": "ownership_gap",
                    "severity": "high",
                    "title": "Ownership gap on active work",
                    "summary": (
                        f"{action_count} action items were detected for only {owner_count} explicit owners. "
                        f"Owner coverage is {owner_coverage:.0%}."
                    ),
                    "confidence": 0.87,
                }
            )

        if action_count > 0 and date_count == 0:
            findings.append(
                {
                    "kind": "timeline_blindspot",
                    "severity": "medium",
                    "title": "Active work without explicit dates",
                    "summary": "The material contains action items but no explicit due dates or milestone dates.",
                    "confidence": 0.78,
                }
            )

        if blocker_count > 0 or dependency_count > 0:
            findings.append(
                {
                    "kind": "delivery_friction",
                    "severity": "high" if blocker_count > 0 else "medium",
                    "title": "Blocked or dependency-heavy delivery path",
                    "summary": (
                        f"{blocker_count} blocker signals and {dependency_count} dependency signals were found "
                        "in the analyzed project material."
                    ),
                    "confidence": 0.81,
                }
            )

        if risk_count >= 2:
            findings.append(
                {
                    "kind": "risk_signal_density",
                    "severity": "medium" if risk_count < 5 else "high",
                    "title": "Repeated risk language across project updates",
                    "summary": (
                        f"{risk_count} lines explicitly mention risk, delay, issues, or schedule concerns."
                    ),
                    "confidence": 0.74,
                }
            )

        all_dates = [date for item in extracted for date in item["dates"]]
        dupes = [date for date, count in Counter(all_dates).items() if count > 3]
        if dupes:
            findings.append(
                {
                    "kind": "milestone_concentration",
                    "severity": "low",
                    "title": "Milestone concentration",
                    "summary": (
                        f"The same dates recur heavily ({', '.join(dupes[:3])}), which can indicate bottlenecks "
                        "or fragile milestone concentration."
                    ),
                    "confidence": 0.61,
                }
            )

        if not findings and retrieved_chunks:
            findings.append(
                {
                    "kind": "manual_review_recommended",
                    "severity": "low",
                    "title": "Manual review recommended",
                    "summary": (
                        f"No strong automated risk was detected for '{objective}', but supporting project evidence "
                        "was retrieved for human review."
                    ),
                    "confidence": 0.48,
                }
            )

        findings.append(
            {
                "kind": "project_risk_score",
                "severity": risk_level,
                "title": f"Project risk score: {risk_score}/100",
                "summary": (
                    f"Computed from ownership coverage, blockers, dependencies, explicit risk mentions, and timeline "
                    f"coverage. Overall risk level is {risk_level}."
                ),
                "confidence": 0.9,
                "score": risk_score,
                "risk_level": risk_level,
                "metrics": {
                    "owners_detected": owner_count,
                    "actions_detected": action_count,
                    "dates_detected": date_count,
                    "risks_detected": risk_count,
                    "blockers_detected": blocker_count,
                    "dependencies_detected": dependency_count,
                    "owner_coverage_ratio": round(owner_coverage, 3),
                },
            }
        )
        return findings

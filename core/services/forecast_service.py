from uuid import uuid4

from db.models.forecast_record import ForecastRecord
from db.models.metric_snapshot import MetricSnapshot


def create_delay_forecast(db, project_ref: str) -> ForecastRecord:
    snapshots = (
        db.query(MetricSnapshot)
        .filter(MetricSnapshot.project_ref == project_ref)
        .order_by(MetricSnapshot.captured_at.desc())
        .all()
    )
    if not snapshots:
        raise ValueError(f"No metric snapshots found for project '{project_ref}'")

    grouped: dict[str, list[MetricSnapshot]] = {}
    for row in snapshots:
        grouped.setdefault(row.metric_name, []).append(row)
    risk_series = []
    for rows in grouped.values():
        risk_series.extend(row.metric_value for row in rows[:4])
    if not risk_series:
        raise ValueError(f"No usable metric history found for project '{project_ref}'")
    scores = list(reversed(risk_series[:8]))
    current_score = scores[-1]
    previous_score = scores[-2] if len(scores) >= 2 else current_score
    trend = round(current_score - previous_score, 1)
    avg_score = sum(scores) / len(scores)

    delay_probability = min(0.97, max(0.05, (avg_score / 100.0) * 0.65 + max(0.0, trend) / 100.0))
    forecast = {
        "current_risk_score": current_score,
        "average_risk_score": round(avg_score, 1),
        "trend_vs_previous": trend,
        "projected_status": "likely_delay" if delay_probability >= 0.6 else "watch" if delay_probability >= 0.35 else "stable",
        "horizon_days": 30,
        "drivers": [f"{name}={rows[0].metric_value}" for name, rows in list(grouped.items())[:5]],
    }

    row = ForecastRecord(
        id=f"forecast_{uuid4().hex[:12]}",
        project_ref=project_ref,
        source=snapshots[0].metric_source,
        delay_probability=round(delay_probability, 3),
        risk_trend=trend,
        forecast=forecast,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row

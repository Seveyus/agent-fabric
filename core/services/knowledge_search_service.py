from db.models.canonical_entity import CanonicalEntity
from db.models.canonical_relation import CanonicalRelation
from db.models.metric_snapshot import MetricSnapshot
from db.models.project_snapshot import ProjectSnapshot


def hybrid_search(db, query: str, project_ref: str | None = None, limit: int = 8) -> list[dict]:
    tokens = [token.lower() for token in query.split() if len(token) > 2]
    if not tokens:
        return []

    node_rows = db.query(CanonicalEntity).all()
    results = []
    for row in node_rows:
        haystack = f"{row.name} {row.entity_ref} {row.attributes}".lower()
        score = sum(1 for token in tokens if token in haystack)
        if score > 0:
            if project_ref and project_ref != row.project_ref and project_ref not in str(row.attributes):
                continue
            results.append(
                {
                    "result_type": "graph_node",
                    "ref": row.entity_ref,
                    "title": row.name,
                    "score": float(score + 2),
                    "summary": str(row.attributes)[:240],
                    "metadata": {"entity_type": row.entity_type},
                }
            )

    edge_rows = db.query(CanonicalRelation).all()
    for row in edge_rows:
        haystack = f"{row.source_ref} {row.target_ref} {row.relation_type} {row.attributes}".lower()
        score = sum(1 for token in tokens if token in haystack)
        if score > 0:
            if project_ref and project_ref not in row.source_ref and project_ref not in row.target_ref:
                continue
            results.append(
                {
                    "result_type": "graph_edge",
                    "ref": f"{row.source_ref}->{row.target_ref}",
                    "title": row.relation_type,
                    "score": float(score + 1),
                    "summary": f"{row.source_ref} -> {row.target_ref}",
                    "metadata": row.attributes,
                }
            )

    metric_rows = db.query(MetricSnapshot).order_by(MetricSnapshot.captured_at.desc()).limit(100).all()
    for row in metric_rows:
        haystack = f"{row.project_ref} {row.metric_source} {row.metric_name} {row.metric_value} {row.dimensions}".lower()
        score = sum(1 for token in tokens if token in haystack)
        if score > 0:
            if project_ref and row.project_ref != project_ref:
                continue
            results.append(
                {
                    "result_type": "metric_snapshot",
                    "ref": row.id,
                    "title": f"{row.project_ref} {row.metric_name}",
                    "score": float(score),
                    "summary": f"{row.metric_name}={row.metric_value}",
                    "metadata": {"project_ref": row.project_ref, "provider": row.metric_source},
                }
            )

    doc_rows = db.query(ProjectSnapshot).order_by(ProjectSnapshot.created_at.desc()).limit(50).all()
    for row in doc_rows:
        haystack = f"{row.run_id} {row.risk_level} {row.risk_score}".lower()
        score = sum(1 for token in tokens if token in haystack)
        if score > 0:
            results.append(
                {
                    "result_type": "document_snapshot",
                    "ref": row.id,
                    "title": f"Run {row.run_id} snapshot",
                    "score": float(score),
                    "summary": f"risk={row.risk_score}/100 level={row.risk_level}",
                    "metadata": {"run_id": row.run_id},
                }
            )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:limit]

def simple_rerank(items: list[dict], query: str) -> list[dict]:
    tokens = {t.lower() for t in query.split() if len(t) > 2}
    scored = []
    for item in items:
        text = item.get("text", "").lower()
        score = sum(1 for t in tokens if t in text)
        scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored]

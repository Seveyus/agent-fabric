from core.llm.router import get_llm


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_llm().embed(texts)

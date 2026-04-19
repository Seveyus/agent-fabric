import json
import httpx

from apps.api.config import settings


class OllamaClient:
    def __init__(self, host: str | None = None):
        self.host = host or settings.ollama_host

    def generate_json(self, prompt: str, schema_hint: dict | None = None) -> dict:
        body = {
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }
        response = httpx.post(f"{self.host}/api/generate", json=body, timeout=180.0)
        response.raise_for_status()
        raw = response.json()["response"]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            response = httpx.post(
                f"{self.host}/api/embeddings",
                json={"model": settings.ollama_embed_model, "prompt": text},
                timeout=120.0,
            )
            response.raise_for_status()
            vectors.append(response.json()["embedding"])
        return vectors

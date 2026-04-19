from typing import Protocol


class LLMClient(Protocol):
    def generate_json(self, prompt: str, schema_hint: dict | None = None) -> dict:
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

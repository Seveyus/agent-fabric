import json
from redis import Redis

from apps.api.config import settings

QUEUE_NAME = "agent_fabric:jobs"


def _client() -> Redis:
    return Redis.from_url(settings.redis_url, decode_responses=True)


def enqueue_pipeline(message: dict) -> None:
    _client().rpush(QUEUE_NAME, json.dumps(message))


def dequeue_pipeline(timeout: int = 2) -> dict | None:
    item = _client().blpop(QUEUE_NAME, timeout=timeout)
    if not item:
        return None
    _, payload = item
    return json.loads(payload)

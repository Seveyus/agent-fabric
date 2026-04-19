import logging

from fastapi import FastAPI

from apps.api.config import settings
from apps.api.middleware.logging import AccessLogMiddleware
from apps.api.middleware.request_id import RequestIdMiddleware
from apps.api.routers import files, health, jobs, pipelines, runs

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title=settings.app_name)

app.add_middleware(RequestIdMiddleware)
app.add_middleware(AccessLogMiddleware)

app.include_router(health.router)
app.include_router(files.router)
app.include_router(jobs.router)
app.include_router(runs.router)
app.include_router(pipelines.router)

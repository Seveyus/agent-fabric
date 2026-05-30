from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from apps.api.deps import get_db
from core.services.dashboard_service import build_dashboard_overview

router = APIRouter(tags=["dashboard"])


@router.get("/", include_in_schema=False)
def dashboard_root():
    return RedirectResponse(url="/dashboard")


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard_page():
    html_path = Path(__file__).resolve().parents[2] / "web" / "dashboard.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@router.get("/dashboard/overview")
def dashboard_overview(db: Session = Depends(get_db)):
    return build_dashboard_overview(db)

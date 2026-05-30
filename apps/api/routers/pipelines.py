from fastapi import APIRouter

router = APIRouter(prefix="/pipelines", tags=["pipelines"])


@router.get("")
def list_pipelines():
    return {
        "pipelines": [
            {
                "name": "project_risk",
                "description": "Detects delivery risk, blockers, ownership gaps, and timeline blindspots."
            }
        ]
    }

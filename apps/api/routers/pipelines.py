from fastapi import APIRouter

router = APIRouter(prefix="/pipelines", tags=["pipelines"])


@router.get("")
def list_pipelines():
    return {
        "pipelines": [
            {
                "name": "project_audit",
                "description": "Detects risks, contradictions, missing ownership and delayed actions."
            }
        ]
    }

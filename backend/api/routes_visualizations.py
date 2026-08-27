from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from backend.api.deps import get_current_session
from backend.session.state import SessionState
from backend.schemas.visualizations import VisualizationRequest, VisualizationResponse
from backend.services.visualization_service import build_visualization

router = APIRouter(prefix="/visualizations", tags=["Visualizations"])


@router.post("/generate", response_model=VisualizationResponse)
def generate_chart(
    req: VisualizationRequest,
    session: SessionState = Depends(get_current_session),
):
    try:
        res = build_visualization(session, req.model_dump())
        return res
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

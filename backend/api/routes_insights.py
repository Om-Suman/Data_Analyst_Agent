from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from backend.api.deps import get_current_session
from backend.session.state import SessionState
from backend.schemas.insights import (
    AIInsightsRequest,
    AIInsightsResponse,
    StatisticalInsightsResponse,
)
from backend.services.analytics_service import (
    get_ai_business_insights,
    get_quick_insights,
)

router = APIRouter(prefix="/insights", tags=["Insights"])


@router.get("/quick", response_model=StatisticalInsightsResponse)
def get_statistical_insights_route(session: SessionState = Depends(get_current_session)):
    try:
        insights = get_quick_insights(session)
        return StatisticalInsightsResponse(insights=insights)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post("/ai", response_model=AIInsightsResponse)
def get_ai_insights_route(
    req: AIInsightsRequest,
    session: SessionState = Depends(get_current_session),
):
    try:
        data = get_ai_business_insights(session, max_tokens=req.max_tokens)
        return AIInsightsResponse(
            executive_summary=data.get("executive_summary", ""),
            key_findings=data.get("key_findings", []),
            trends=data.get("trends", []),
            opportunities=data.get("opportunities", []),
            risks=data.get("risks", []),
            recommendations=data.get("recommendations", []),
            data_story=data.get("data_story", ""),
            cached=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get("/ai/cached", response_model=AIInsightsResponse)
def get_cached_ai_insights(session: SessionState = Depends(get_current_session)):
    data = session.last_insights or {}
    return AIInsightsResponse(
        executive_summary=data.get("executive_summary", ""),
        key_findings=data.get("key_findings", []),
        trends=data.get("trends", []),
        opportunities=data.get("opportunities", []),
        risks=data.get("risks", []),
        recommendations=data.get("recommendations", []),
        data_story=data.get("data_story", ""),
        cached=True,
    )

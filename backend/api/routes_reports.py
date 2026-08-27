from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status
from backend.api.deps import get_current_session
from backend.session.state import SessionState
from backend.schemas.reports import HTMLReportRequest, ProfileResponse
from backend.services.report_service import (
    generate_excel_report,
    generate_html_report,
    generate_quick_profile,
)

router = APIRouter(prefix="/reports", tags=["Reports"])


@router.post("/html")
def get_html_report_route(
    req: HTMLReportRequest,
    session: SessionState = Depends(get_current_session),
):
    try:
        html_content = generate_html_report(
            session=session,
            include_sample=req.include_sample,
            include_stats=req.include_stats,
            include_insights=req.include_insights,
        )
        return Response(
            content=html_content,
            media_type="text/html",
            headers={"Content-Disposition": f'attachment; filename="{session.active_dataset or "dataset"}_report.html"'},
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get("/excel")
def get_excel_report_route(session: SessionState = Depends(get_current_session)):
    try:
        excel_bytes = generate_excel_report(session)
        return Response(
            content=excel_bytes,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{session.active_dataset or "dataset"}_report.xlsx"'},
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get("/profile", response_model=ProfileResponse)
def get_data_profile_route(session: SessionState = Depends(get_current_session)):
    try:
        return generate_quick_profile(session)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from backend.api.deps import get_current_session
from backend.session.state import SessionState
from backend.schemas.cleaning import (
    CleaningConfigRequest,
    CleaningReportResponse,
    ColumnTransformRequest,
    ColumnTransformResponse,
    QualityScoreResponse,
    RollbackRequest,
    VersionHistoryResponse,
)
from backend.services.cleaning_service import (
    apply_cleaning,
    execute_cleaning_dry_run,
    get_dataset_quality_overview,
    get_version_history,
    rollback_dataset_version,
    transform_column,
)

router = APIRouter(prefix="/cleaning", tags=["Data Cleaning"])


@router.get("/quality", response_model=QualityScoreResponse)
def get_quality_overview(session: SessionState = Depends(get_current_session)):
    try:
        return get_dataset_quality_overview(session)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post("/preview", response_model=CleaningReportResponse)
def preview_cleaning(
    req: CleaningConfigRequest,
    session: SessionState = Depends(get_current_session),
):
    try:
        _, _, report_data = execute_cleaning_dry_run(session, req.model_dump())
        return report_data
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post("/apply", response_model=CleaningReportResponse)
def run_and_apply_cleaning(
    req: CleaningConfigRequest,
    session: SessionState = Depends(get_current_session),
):
    try:
        report_data = apply_cleaning(session, req.model_dump())
        return report_data
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post("/transform-column", response_model=ColumnTransformResponse)
def apply_column_transformation(
    req: ColumnTransformRequest,
    session: SessionState = Depends(get_current_session),
):
    try:
        res = transform_column(session, req.model_dump())
        return res
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get("/versions", response_model=VersionHistoryResponse)
def list_versions(session: SessionState = Depends(get_current_session)):
    try:
        return get_version_history(session)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post("/rollback")
def rollback_version(
    req: RollbackRequest,
    session: SessionState = Depends(get_current_session),
):
    try:
        rollback_dataset_version(session, req.version)
        return {"status": "ok", "message": f"Successfully rolled back to version {req.version}"}
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

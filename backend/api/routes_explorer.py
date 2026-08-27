from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from backend.api.deps import get_current_session
from backend.session.state import SessionState
from backend.schemas.explorer import (
    ColumnProfileResponse,
    CorrelationsRequest,
    CorrelationsResponse,
    DistributionRequest,
    DistributionResponse,
    ExplorerBrowseRequest,
    ExplorerBrowseResponse,
)
from backend.services.explorer_service import (
    browse_dataset,
    compute_column_profile,
    compute_correlations,
    compute_distribution,
)

router = APIRouter(prefix="/explorer", tags=["Data Explorer"])


@router.post("/browse", response_model=ExplorerBrowseResponse)
def browse(
    req: ExplorerBrowseRequest,
    session: SessionState = Depends(get_current_session),
):
    try:
        filters_dict = [f.model_dump() for f in req.filters] if req.filters else None
        return browse_dataset(
            session=session,
            columns=req.columns,
            page=req.page,
            page_size=req.page_size,
            sort_by=req.sort_by,
            sort_dir=req.sort_dir,
            filters=filters_dict,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post("/correlations", response_model=CorrelationsResponse)
def get_correlations(
    req: CorrelationsRequest,
    session: SessionState = Depends(get_current_session),
):
    try:
        return compute_correlations(session, columns=req.columns, method=req.method)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post("/distribution", response_model=DistributionResponse)
def get_distribution(
    req: DistributionRequest,
    session: SessionState = Depends(get_current_session),
):
    try:
        return compute_distribution(
            session=session,
            column=req.column,
            chart_type=req.chart_type,
            group_by=req.group_by,
            nbins=req.nbins,
            top_n=req.top_n,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get("/profile/{column}", response_model=ColumnProfileResponse)
def get_column_profile_data(
    column: str,
    session: SessionState = Depends(get_current_session),
):
    try:
        return compute_column_profile(session, column=column)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

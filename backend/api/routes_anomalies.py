from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status
import pandas as pd
from backend.api.deps import get_current_session
from backend.session.state import SessionState
from backend.schemas.anomalies import AnomalyRequest, AnomalyResponse
from backend.services.analytics_service import run_anomaly_detection

router = APIRouter(prefix="/anomalies", tags=["Anomaly Detection"])


@router.post("/run", response_model=AnomalyResponse)
def execute_anomaly_detection_route(
    req: AnomalyRequest,
    session: SessionState = Depends(get_current_session),
):
    try:
        return run_anomaly_detection(
            session=session,
            method=req.method,
            contamination=req.contamination,
            threshold=req.threshold,
            factor=req.factor,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get("/download")
def download_cached_anomalies(session: SessionState = Depends(get_current_session)):
    res = session.cached_anomaly
    if not res or res.anomaly_df is None or res.anomaly_df.empty:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No anomalous records available to download.")

    csv_bytes = res.anomaly_df.to_csv(index=False).encode("utf-8")
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="anomalies.csv"'},
    )

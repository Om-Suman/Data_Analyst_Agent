from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status
import pandas as pd
from backend.api.deps import get_current_session
from backend.session.state import SessionState
from backend.schemas.forecasting import ForecastingRequest, ForecastingResponse
from backend.services.analytics_service import run_forecast

router = APIRouter(prefix="/forecasting", tags=["Forecasting"])


@router.post("/run", response_model=ForecastingResponse)
def execute_forecast_route(
    req: ForecastingRequest,
    session: SessionState = Depends(get_current_session),
):
    try:
        return run_forecast(
            session=session,
            target_col=req.target_col,
            method=req.method,
            horizon=req.horizon,
            window=req.window,
            alpha=req.alpha,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get("/download")
def download_cached_forecast(session: SessionState = Depends(get_current_session)):
    fc = session.cached_forecast
    if not fc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No forecast available to download.")

    df = pd.DataFrame({
        "Period": fc.forecast_index,
        "Forecast": fc.forecast_values,
        "Lower_95_CI": fc.confidence_lower,
        "Upper_95_CI": fc.confidence_upper,
    })
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="forecast.csv"'},
    )

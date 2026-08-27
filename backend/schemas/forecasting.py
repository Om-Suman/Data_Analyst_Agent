from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


class ForecastingRequest(BaseModel):
    target_col: str
    method: str = Field(default="Moving Average", description="Moving Average|Linear Trend|Exponential Smoothing")
    horizon: int = Field(default=30, ge=1, le=365)
    window: int = Field(default=7, ge=2, le=90)
    alpha: float = Field(default=0.3, ge=0.01, le=1.0)


class ForecastingResponse(BaseModel):
    method: str
    column: str
    horizon: int
    forecast_index: list[Any]
    forecast_values: list[float]
    confidence_lower: list[float]
    confidence_upper: list[float]
    metrics: dict[str, Any] = {}
    interpretation: str
    figure_spec: Optional[dict[str, Any]] = None
    historical_points: int = 0

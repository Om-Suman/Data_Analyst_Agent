from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


class AnomalyRequest(BaseModel):
    method: str = Field(default="Isolation Forest", description="Isolation Forest|Z-Score|IQR")
    contamination: float = Field(default=0.05, ge=0.01, le=0.5)
    threshold: float = Field(default=3.0, ge=1.0, le=10.0)
    factor: float = Field(default=1.5, ge=0.5, le=5.0)


class AnomalyResponse(BaseModel):
    method: str
    n_anomalies: int
    anomaly_rate: float
    anomaly_indices: list[int]
    anomalous_rows: list[dict[str, Any]]
    scores: list[float]
    columns_used: list[str]
    figure_spec: Optional[dict[str, Any]] = None
    recommendations: list[str] = []

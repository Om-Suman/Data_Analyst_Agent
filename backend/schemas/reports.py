from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class HTMLReportRequest(BaseModel):
    include_sample: bool = Field(default=True)
    include_stats: bool = Field(default=True)
    include_insights: bool = Field(default=True)


class ProfileResponse(BaseModel):
    dataset_name: str
    rows: int
    cols: int
    numeric_columns: list[dict[str, Any]]
    categorical_columns: list[dict[str, Any]]

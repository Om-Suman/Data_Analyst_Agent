from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


class DatasetItem(BaseModel):
    name: str
    source: str
    uploaded_at: str
    rows: int
    cols: int
    version: int
    is_active: bool
    is_text: bool


class DatasetListResponse(BaseModel):
    datasets: list[DatasetItem]
    active_dataset: Optional[str] = None


class DatasetPreviewResponse(BaseModel):
    name: str
    rows: int
    cols: int
    columns: list[str]
    data: list[dict[str, Any]]
    metadata: dict[str, Any]
    numeric_cols: list[str]
    categorical_cols: list[str]
    date_cols: list[str]
    is_text: bool = False
    text_content: Optional[str] = None


class SampleDatasetRequest(BaseModel):
    sample_name: str = Field(description="Name of sample: 'Sales Data', 'Employee Data', 'Finance Data'")


class SetActiveRequest(BaseModel):
    name: str

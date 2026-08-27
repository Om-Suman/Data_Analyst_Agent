from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


class FilterItem(BaseModel):
    column: str
    operator: str = Field(default="in", description="in|range|equals")
    values: Optional[list[Any]] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None


class ExplorerBrowseRequest(BaseModel):
    columns: Optional[list[str]] = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=1000)
    sort_by: Optional[str] = None
    sort_dir: Optional[str] = Field(default="asc", description="asc|desc")
    filters: Optional[list[FilterItem]] = None


class ExplorerBrowseResponse(BaseModel):
    total_rows: int
    total_unfiltered_rows: int
    page: int
    page_size: int
    columns: list[str]
    data: list[dict[str, Any]]
    filters_applied: list[str] = []


class CorrelationsRequest(BaseModel):
    columns: list[str]
    method: str = Field(default="pearson", description="pearson|spearman|kendall")


class CorrelationPair(BaseModel):
    col_a: str
    col_b: str
    correlation: float


class CorrelationsResponse(BaseModel):
    matrix: dict[str, dict[str, float]]
    columns: list[str]
    top_pairs: list[CorrelationPair]
    figure_spec: Optional[dict[str, Any]] = None


class DistributionRequest(BaseModel):
    column: str
    chart_type: str = Field(default="Histogram", description="Histogram|Box Plot|Violin")
    group_by: Optional[str] = None
    nbins: int = 30
    top_n: int = 15


class DistributionResponse(BaseModel):
    column: str
    chart_type: str
    figure_spec: dict[str, Any]


class ColumnProfileResponse(BaseModel):
    column: str
    dtype: str
    total_count: int
    missing_count: int
    missing_pct: float
    unique_count: int
    is_numeric: bool
    numeric_stats: Optional[dict[str, Any]] = None
    top_values: Optional[list[dict[str, Any]]] = None
    figure_spec: Optional[dict[str, Any]] = None

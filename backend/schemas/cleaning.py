from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


class CleaningConfigRequest(BaseModel):
    missing_strategy: str = Field(default="mean", description="mean|median|mode|ffill|bfill|drop_rows|drop_cols|custom|none")
    custom_fill_value: Optional[Any] = None
    remove_duplicates: bool = True
    fix_dtypes: bool = True
    normalize_column_names: bool = True
    outlier_method: str = Field(default="none", description="none|zscore|iqr")
    outlier_threshold: float = 3.0
    iqr_factor: float = 1.5
    columns_to_clean: Optional[list[str]] = None


class CleaningReportResponse(BaseModel):
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    duplicates_removed: int
    missing_filled: dict[str, Any]
    cols_dropped: list[str]
    outliers_removed: int
    dtype_changes: dict[str, str]
    col_renames: dict[str, str]
    quality_score_before: float
    quality_score_after: float
    quality_grade_before: str
    quality_grade_after: str
    recommendations: list[str]
    preview_data: Optional[list[dict[str, Any]]] = None


class QualityScoreResponse(BaseModel):
    quality_score: float
    quality_grade: str
    missing_total: int
    missing_pct: float
    duplicate_rows: int
    duplicate_pct: float
    missing_by_column: list[dict[str, Any]]
    dtypes_by_column: list[dict[str, Any]]


class VersionItem(BaseModel):
    version: int
    timestamp: str
    rows: int
    cols: int
    description: str


class VersionHistoryResponse(BaseModel):
    dataset_name: str
    current_version: int
    versions: list[VersionItem]


class RollbackRequest(BaseModel):
    version: int

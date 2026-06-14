"""
Data Cleaning Module
Handles missing values, duplicates, outliers, type corrections, and quality scoring.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CleaningConfig:
    missing_strategy: str = "mean"          # mean|median|mode|ffill|bfill|drop_rows|drop_cols|custom
    custom_fill_value: object = None
    remove_duplicates: bool = True
    fix_dtypes: bool = True
    normalize_column_names: bool = True
    outlier_method: str = "none"            # none|zscore|iqr
    outlier_threshold: float = 3.0          # z-score threshold
    iqr_factor: float = 1.5
    columns_to_clean: Optional[list] = None  # None = all


@dataclass
class CleaningReport:
    rows_before: int = 0
    rows_after: int = 0
    cols_before: int = 0
    cols_after: int = 0
    duplicates_removed: int = 0
    missing_filled: dict = field(default_factory=dict)
    cols_dropped: list = field(default_factory=list)
    outliers_removed: int = 0
    dtype_changes: dict = field(default_factory=dict)
    col_renames: dict = field(default_factory=dict)
    quality_score_before: float = 0.0
    quality_score_after: float = 0.0
    quality_grade_before: str = ""
    quality_grade_after: str = ""
    recommendations: list = field(default_factory=list)


def compute_quality_score(df: pd.DataFrame) -> tuple[float, str]:
    """Score: 0-100, grade: A/B/C/D/F"""
    score = 100.0
    total_cells = df.shape[0] * df.shape[1]
    if total_cells == 0:
        return 0.0, "F"

    # Missing values penalty (up to 40 pts)
    missing_ratio = df.isnull().sum().sum() / total_cells
    score -= missing_ratio * 40

    # Duplicate penalty (up to 20 pts)
    dup_ratio = df.duplicated().sum() / max(len(df), 1)
    score -= dup_ratio * 20

    # Mixed dtype penalty (up to 20 pts)
    mixed = 0
    for col in df.select_dtypes(include="object").columns:
        types = df[col].dropna().map(type).nunique()
        if types > 1:
            mixed += 1
    score -= (mixed / max(len(df.columns), 1)) * 20

    score = max(0.0, min(100.0, score))
    if score >= 90:
        grade = "A"
    elif score >= 75:
        grade = "B"
    elif score >= 60:
        grade = "C"
    elif score >= 40:
        grade = "D"
    else:
        grade = "F"
    return round(score, 1), grade


def normalize_column_names(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Standardize column names to snake_case."""
    import re
    renames = {}
    for col in df.columns:
        new_col = col.strip()
        new_col = re.sub(r"[^\w\s]", "", new_col)
        new_col = re.sub(r"\s+", "_", new_col)
        new_col = new_col.lower()
        if new_col != col:
            renames[col] = new_col
    if renames:
        df = df.rename(columns=renames)
    return df, renames


def fill_missing(df: pd.DataFrame, strategy: str, custom_value=None, columns=None) -> tuple[pd.DataFrame, dict]:
    """Fill missing values per strategy."""
    cols = columns or df.columns.tolist()
    filled = {}

    for col in cols:
        n_missing = df[col].isnull().sum()
        if n_missing == 0:
            continue

        if strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
        elif strategy == "ffill":
            df[col] = df[col].ffill()
        elif strategy == "bfill":
            df[col] = df[col].bfill()
        elif strategy == "drop_rows":
            df = df.dropna(subset=[col])
        elif strategy == "drop_cols":
            df = df.drop(columns=[col])
            filled[col] = "dropped"
            continue
        elif strategy == "custom" and custom_value is not None:
            df[col] = df[col].fillna(custom_value)

        new_missing = df[col].isnull().sum() if col in df.columns else 0
        filled[col] = int(n_missing - new_missing)

    return df, filled


def detect_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.Series:
    numeric = df.select_dtypes(include=np.number)
    z = np.abs((numeric - numeric.mean()) / numeric.std(ddof=0))
    return (z > threshold).any(axis=1)


def detect_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> pd.Series:
    numeric = df.select_dtypes(include=np.number)
    Q1 = numeric.quantile(0.25)
    Q3 = numeric.quantile(0.75)
    IQR = Q3 - Q1
    mask = ((numeric < Q1 - factor * IQR) | (numeric > Q3 + factor * IQR)).any(axis=1)
    return mask


def clean_dataframe(df: pd.DataFrame, config: CleaningConfig) -> tuple[pd.DataFrame, CleaningReport]:
    report = CleaningReport()
    report.rows_before = len(df)
    report.cols_before = len(df.columns)
    report.quality_score_before, report.quality_grade_before = compute_quality_score(df)

    df = df.copy()

    # 1. Normalize column names
    if config.normalize_column_names:
        df, renames = normalize_column_names(df)
        report.col_renames = renames

    # 2. Fix dtypes
    if config.fix_dtypes:
        for col in df.columns:
            if df[col].dtype == object:
                # Try numeric
                try:
                    conv = pd.to_numeric(df[col], errors="coerce")
                    if conv.notna().sum() / max(len(df), 1) > 0.9:
                        original = df[col].dtype
                        df[col] = conv
                        report.dtype_changes[col] = f"{original} → {df[col].dtype}"
                        continue
                except Exception:
                    pass
                # Try datetime
                try:
                    conv = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                    if conv.notna().sum() / max(len(df), 1) > 0.9:
                        original = df[col].dtype
                        df[col] = conv
                        report.dtype_changes[col] = f"{original} → {df[col].dtype}"
                except Exception:
                    pass

    # 3. Remove duplicates
    if config.remove_duplicates:
        n_before = len(df)
        df = df.drop_duplicates()
        report.duplicates_removed = n_before - len(df)

    # 4. Handle missing values
    if config.missing_strategy != "none":
        df, filled = fill_missing(df, config.missing_strategy, config.custom_fill_value, config.columns_to_clean)
        report.missing_filled = filled
        if config.missing_strategy == "drop_cols":
            report.cols_dropped = [c for c, v in filled.items() if v == "dropped"]

    # 5. Outlier removal
    if config.outlier_method == "zscore":
        mask = detect_outliers_zscore(df, config.outlier_threshold)
        n_before = len(df)
        df = df[~mask]
        report.outliers_removed = n_before - len(df)
    elif config.outlier_method == "iqr":
        mask = detect_outliers_iqr(df, config.iqr_factor)
        n_before = len(df)
        df = df[~mask]
        report.outliers_removed = n_before - len(df)

    report.rows_after = len(df)
    report.cols_after = len(df.columns)
    report.quality_score_after, report.quality_grade_after = compute_quality_score(df)

    # Recommendations
    if report.quality_score_after < 80:
        report.recommendations.append("Consider investigating remaining missing values.")
    if report.outliers_removed == 0 and config.outlier_method == "none":
        report.recommendations.append("Run outlier detection to identify anomalous rows.")
    if report.duplicates_removed > 0:
        report.recommendations.append(f"Removed {report.duplicates_removed} duplicates — check if deduplication was appropriate.")

    return df, report

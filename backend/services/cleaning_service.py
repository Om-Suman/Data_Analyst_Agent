from __future__ import annotations

from typing import Any, Optional
import pandas as pd
import numpy as np

from backend.session.state import SessionState
from backend.services.dataset_service import sanitize_dataframe_for_json, to_json_compatible
from modules.cleaning import (
    CleaningConfig,
    CleaningReport,
    clean_dataframe,
    compute_quality_score,
)


def get_dataset_quality_overview(session: SessionState) -> dict[str, Any]:
    df = session.get_active_df()
    if df is None:
        raise ValueError("No active dataset to assess quality.")

    score, grade = compute_quality_score(df)
    missing_total = int(df.isnull().sum().sum())
    total_cells = max(len(df) * len(df.columns), 1)
    missing_pct = round(missing_total / total_cells * 100, 2)

    dup_count = int(df.duplicated().sum())
    dup_pct = round(dup_count / max(len(df), 1) * 100, 2)

    missing_by_col = []
    missing_counts = df.isnull().sum()
    for col in df.columns:
        if missing_counts[col] > 0:
            missing_by_col.append({
                "column": col,
                "missing_count": int(missing_counts[col]),
                "missing_pct": round(float(missing_counts[col]) / max(len(df), 1) * 100, 2),
            })

    dtypes_by_col = []
    for col in df.columns:
        dtypes_by_col.append({
            "column": col,
            "dtype": str(df[col].dtype),
        })

    return {
        "quality_score": float(score),
        "quality_grade": grade,
        "missing_total": missing_total,
        "missing_pct": missing_pct,
        "duplicate_rows": dup_count,
        "duplicate_pct": dup_pct,
        "missing_by_column": missing_by_col,
        "dtypes_by_column": dtypes_by_col,
    }


def execute_cleaning_dry_run(
    session: SessionState,
    config_dict: dict[str, Any],
) -> tuple[pd.DataFrame, CleaningReport, dict[str, Any]]:
    df = session.get_active_df()
    if df is None:
        raise ValueError("No active dataset to clean.")

    config = CleaningConfig(
        missing_strategy=config_dict.get("missing_strategy", "mean"),
        custom_fill_value=config_dict.get("custom_fill_value"),
        remove_duplicates=config_dict.get("remove_duplicates", True),
        fix_dtypes=config_dict.get("fix_dtypes", True),
        normalize_column_names=config_dict.get("normalize_column_names", True),
        outlier_method=config_dict.get("outlier_method", "none"),
        outlier_threshold=float(config_dict.get("outlier_threshold", 3.0)),
        iqr_factor=float(config_dict.get("iqr_factor", 1.5)),
        columns_to_clean=config_dict.get("columns_to_clean"),
    )

    cleaned_df, report = clean_dataframe(df, config)
    preview = sanitize_dataframe_for_json(cleaned_df.head(20))

    report_data = {
        "rows_before": int(report.rows_before),
        "rows_after": int(report.rows_after),
        "cols_before": int(report.cols_before),
        "cols_after": int(report.cols_after),
        "duplicates_removed": int(report.duplicates_removed),
        "missing_filled": report.missing_filled,
        "cols_dropped": report.cols_dropped,
        "outliers_removed": int(report.outliers_removed),
        "dtype_changes": report.dtype_changes,
        "col_renames": report.col_renames,
        "quality_score_before": float(report.quality_score_before),
        "quality_score_after": float(report.quality_score_after),
        "quality_grade_before": report.quality_grade_before,
        "quality_grade_after": report.quality_grade_after,
        "recommendations": report.recommendations,
        "preview_data": preview,
    }
    report_data = to_json_compatible(report_data)

    return cleaned_df, report, report_data


def apply_cleaning(
    session: SessionState,
    config_dict: dict[str, Any],
) -> dict[str, Any]:
    name = session.active_dataset
    if not name or name not in session.datasets:
        raise ValueError("No active dataset to clean.")

    cleaned_df, report, report_data = execute_cleaning_dry_run(session, config_dict)

    session.save_version(name, f"Cleaned ({config_dict.get('missing_strategy', 'default')})")
    session.datasets[name]["df"] = cleaned_df
    session.datasets[name]["rows"] = int(len(cleaned_df))
    session.datasets[name]["cols"] = int(len(cleaned_df.columns))
    session.cleaning_log.append({
        "dataset": name,
        "config": config_dict,
        "report": report_data,
    })

    return report_data


def get_version_history(session: SessionState) -> dict[str, Any]:
    name = session.active_dataset
    if not name or name not in session.datasets:
        raise ValueError("No active dataset selected.")

    record = session.datasets[name]
    versions = session.dataset_versions.get(name, [])
    items = []
    for v in reversed(versions):
        ts = v["timestamp"].isoformat() if hasattr(v["timestamp"], "isoformat") else str(v["timestamp"])
        items.append({
            "version": int(v["version"]),
            "timestamp": ts,
            "rows": int(v["rows"]),
            "cols": int(v["cols"]),
            "description": str(v.get("description", "")),
        })

    return {
        "dataset_name": name,
        "current_version": int(record.get("version", 1)),
        "versions": items,
    }


def rollback_dataset_version(session: SessionState, version_number: int) -> bool:
    name = session.active_dataset
    if not name:
        raise ValueError("No active dataset selected.")
    success = session.rollback_version(name, version_number)
    if not success:
        raise ValueError(f"Version {version_number} not found for dataset '{name}'.")
    return True


def transform_column(session: SessionState, req: dict[str, Any]) -> dict[str, Any]:
    name = session.active_dataset
    if not name or name not in session.datasets:
        raise ValueError("No active dataset selected.")

    df = session.get_active_df()
    if df is None:
        raise ValueError("Active dataset DataFrame not found.")

    df_copy = df.copy()
    op = req.get("operation")
    col = req.get("column")
    target_type = req.get("target_type")
    new_name = req.get("new_name")
    expr = req.get("expression")
    case_mode = req.get("case_mode")

    msg = ""

    if op == "rename":
        if not new_name or not new_name.strip():
            raise ValueError("New column name is required for rename.")
        if col not in df_copy.columns:
            raise ValueError(f"Column '{col}' not found.")
        df_copy.rename(columns={col: new_name.strip()}, inplace=True)
        msg = f"Renamed column '{col}' to '{new_name.strip()}'"

    elif op == "drop":
        if col not in df_copy.columns:
            raise ValueError(f"Column '{col}' not found.")
        df_copy.drop(columns=[col], inplace=True)
        msg = f"Dropped column '{col}'"

    elif op == "cast":
        if col not in df_copy.columns:
            raise ValueError(f"Column '{col}' not found.")
        if target_type == "int":
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce").fillna(0).astype(int)
        elif target_type == "float":
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
        elif target_type == "datetime":
            df_copy[col] = pd.to_datetime(df_copy[col], errors="coerce")
        elif target_type == "string":
            df_copy[col] = df_copy[col].astype(str)
        elif target_type == "boolean":
            df_copy[col] = df_copy[col].astype(bool)
        elif target_type == "category":
            df_copy[col] = df_copy[col].astype("category")
        else:
            raise ValueError(f"Unsupported target type '{target_type}'")
        msg = f"Converted column '{col}' dtype to {target_type}"

    elif op == "string_case":
        if col not in df_copy.columns:
            raise ValueError(f"Column '{col}' not found.")
        str_series = df_copy[col].astype(str)
        if case_mode == "upper":
            df_copy[col] = str_series.str.upper()
        elif case_mode == "lower":
            df_copy[col] = str_series.str.lower()
        elif case_mode == "title":
            df_copy[col] = str_series.str.title()
        elif case_mode == "trim":
            df_copy[col] = str_series.str.strip()
        else:
            raise ValueError(f"Unsupported case mode '{case_mode}'")
        msg = f"Applied '{case_mode}' string transformation to '{col}'"

    elif op in ("math_expr", "create_column"):
        if not expr or not expr.strip():
            raise ValueError("Expression is required.")
        # Safe evaluation in isolated namespace with pandas Series
        safe_dict = {c: df_copy[c] for c in df_copy.columns}
        safe_dict["np"] = np
        safe_dict["pd"] = pd
        # Target column is col if specified, or new_name
        dest_col = col if (col and op == "math_expr") else (new_name or col or "new_col")
        try:
            result_series = eval(expr, {"__builtins__": {}}, safe_dict)
            df_copy[dest_col] = result_series
            msg = f"Evaluated expression '{expr}' into column '{dest_col}'"
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expr}': {str(e)}")
    else:
        raise ValueError(f"Unknown operation '{op}'")

    session.update_dataset(name, df_copy, description=msg)
    preview = sanitize_dataframe_for_json(df_copy.head(20))

    return {
        "success": True,
        "operation": op,
        "message": msg,
        "rows": len(df_copy),
        "cols": len(df_copy.columns),
        "columns": list(df_copy.columns),
        "sample_preview": preview,
    }

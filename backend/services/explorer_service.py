from __future__ import annotations

import json
from typing import Any, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from backend.session.state import SessionState
from backend.services.dataset_service import sanitize_dataframe_for_json, to_json_compatible


def browse_dataset(
    session: SessionState,
    columns: Optional[list[str]] = None,
    page: int = 1,
    page_size: int = 50,
    sort_by: Optional[str] = None,
    sort_dir: Optional[str] = "asc",
    filters: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    df = session.get_active_df()
    if df is None:
        raise ValueError("No active dataset.")

    filtered_df = df.copy()
    filters_applied = []

    if filters:
        for f in filters:
            col = f.get("column")
            if not col or col not in filtered_df.columns:
                continue
            op = f.get("operator", "in")
            if op == "in" and f.get("values"):
                vals = f["values"]
                filtered_df = filtered_df[filtered_df[col].isin(vals)]
                filters_applied.append(f"{col} in {vals}")
            elif op == "range":
                min_v = f.get("min_val")
                max_v = f.get("max_val")
                if min_v is not None:
                    filtered_df = filtered_df[filtered_df[col] >= min_v]
                if max_v is not None:
                    filtered_df = filtered_df[filtered_df[col] <= max_v]
                filters_applied.append(f"{col} ∈ [{min_v}, {max_v}]")

    if sort_by and sort_by in filtered_df.columns:
        ascending = (sort_dir != "desc")
        filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)

    total_rows = len(filtered_df)
    target_columns = [c for c in (columns or df.columns.tolist()) if c in filtered_df.columns]
    if not target_columns:
        target_columns = df.columns.tolist()

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_slice = filtered_df[target_columns].iloc[start_idx:end_idx]

    data = sanitize_dataframe_for_json(page_slice)

    return {
        "total_rows": total_rows,
        "total_unfiltered_rows": len(df),
        "page": page,
        "page_size": page_size,
        "columns": target_columns,
        "data": data,
        "filters_applied": filters_applied,
    }


def compute_correlations(
    session: SessionState,
    columns: list[str],
    method: str = "pearson",
) -> dict[str, Any]:
    df = session.get_active_df()
    if df is None:
        raise ValueError("No active dataset.")

    numeric_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        raise ValueError("Need at least 2 numeric columns for correlation analysis.")

    corr_df = df[numeric_cols].corr(method=method).round(4)
    # Replace NaN values with 0.0 to prevent validation errors
    corr_df = corr_df.fillna(0.0)
    matrix = {str(col): {str(k): to_json_compatible(v) for k, v in corr_df[col].to_dict().items()} for col in corr_df.columns}

    # Top pairs
    top_pairs = []
    for i in range(len(corr_df.columns)):
        for j in range(i + 1, len(corr_df.columns)):
            col_a = str(corr_df.columns[i])
            col_b = str(corr_df.columns[j])
            val = float(corr_df.iloc[i, j])
            if not np.isnan(val):
                top_pairs.append({
                    "col_a": col_a,
                    "col_b": col_b,
                    "correlation": round(val, 4),
                    "strength": abs(val),
                })
    top_pairs.sort(key=lambda x: x["strength"], reverse=True)

    # Plotly heatmap spec
    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        template="plotly_dark",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=f"{method.title()} Correlation Matrix",
    )
    fig.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
    fig_json = json.loads(fig.to_json())

    clean_pairs = [{"col_a": p["col_a"], "col_b": p["col_b"], "correlation": p["correlation"]} for p in top_pairs[:20]]

    return to_json_compatible({
        "matrix": matrix,
        "columns": numeric_cols,
        "top_pairs": clean_pairs,
        "figure_spec": fig_json,
    })


def compute_distribution(
    session: SessionState,
    column: str,
    chart_type: str = "Histogram",
    group_by: Optional[str] = None,
    nbins: int = 30,
    top_n: int = 15,
) -> dict[str, Any]:
    df = session.get_active_df()
    if df is None:
        raise ValueError("No active dataset.")
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    col_data = df[column]
    color = group_by if group_by and group_by in df.columns and group_by != "None" else None

    if pd.api.types.is_numeric_dtype(col_data):
        if chart_type == "Histogram":
            fig = px.histogram(
                df, x=column, nbins=nbins, color=color,
                template="plotly_dark", color_discrete_sequence=["#4f8ef7"], marginal="box",
                title=f"Distribution of {column}",
            )
        elif chart_type == "Box Plot":
            fig = px.box(
                df, y=column, color=color, template="plotly_dark", points="outliers",
                title=f"Box Plot of {column}",
            )
        else:
            fig = px.violin(
                df, y=column, color=color, template="plotly_dark", box=True,
                title=f"Violin Plot of {column}",
            )
    else:
        vc = df[column].value_counts().head(top_n).reset_index()
        vc.columns = [column, "Count"]
        fig = px.bar(
            vc, x=column, y="Count", template="plotly_dark",
            color="Count", color_continuous_scale="Blues", title=f"Top {top_n}: {column}",
        )

    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=400)
    return {
        "column": column,
        "chart_type": chart_type,
        "figure_spec": json.loads(fig.to_json()),
    }


def compute_column_profile(session: SessionState, column: str) -> dict[str, Any]:
    df = session.get_active_df()
    if df is None:
        raise ValueError("No active dataset.")
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    col_data = df[column]
    is_num = pd.api.types.is_numeric_dtype(col_data)
    total_len = max(len(df), 1)
    missing_cnt = int(col_data.isnull().sum())
    missing_pct = round(missing_cnt / total_len * 100, 2)
    unique_cnt = int(col_data.nunique())

    numeric_stats = None
    top_values = None
    fig_spec = None

    if is_num:
        clean = col_data.dropna()
        numeric_stats = {
            "min": float(clean.min()) if not clean.empty else None,
            "max": float(clean.max()) if not clean.empty else None,
            "mean": round(float(clean.mean()), 4) if not clean.empty else None,
            "std": round(float(clean.std()), 4) if not clean.empty else None,
            "median": round(float(clean.median()), 4) if not clean.empty else None,
            "skew": round(float(clean.skew()), 3) if not clean.empty else None,
            "kurtosis": round(float(clean.kurtosis()), 3) if not clean.empty else None,
            "zeros": int((clean == 0).sum()),
        }
        fig = px.histogram(
            df, x=column, nbins=40, template="plotly_dark",
            color_discrete_sequence=["#4f8ef7"], marginal="rug",
            title=f"Histogram: {column}",
        )
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        fig_spec = json.loads(fig.to_json())
    else:
        vc = col_data.value_counts().head(20).reset_index()
        vc.columns = ["value", "count"]
        vc["percent"] = (vc["count"] / total_len * 100).round(1)
        top_values = vc.to_dict(orient="records")

        fig = px.bar(
            vc, x="value", y="count", template="plotly_dark",
            color_discrete_sequence=["#4f8ef7"], title=f"Top Categories: {column}",
        )
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        fig_spec = json.loads(fig.to_json())

    return to_json_compatible({
        "column": str(column),
        "dtype": str(col_data.dtype),
        "total_count": int(len(col_data)),
        "missing_count": int(missing_cnt),
        "missing_pct": float(missing_pct),
        "unique_count": int(unique_cnt),
        "is_numeric": bool(is_num),
        "numeric_stats": numeric_stats,
        "top_values": top_values,
        "figure_spec": fig_spec,
    })

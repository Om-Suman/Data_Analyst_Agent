from __future__ import annotations

import json
from typing import Any, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from backend.session.state import SessionState


def _positive_boolean_mask(series: pd.Series) -> pd.Series | None:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    normalized = series.dropna().astype(str).str.strip().str.lower()
    if normalized.empty:
        return None

    positive_values = {"yes", "y", "true", "t", "1", "has_chip", "chip", "chipped"}
    negative_values = {"no", "n", "false", "f", "0", "none", "nan", ""}
    allowed_values = positive_values | negative_values

    if normalized.isin(allowed_values).all() and normalized.nunique() <= 2:
        full_normalized = series.astype(str).str.strip().str.lower()
        return full_normalized.isin(positive_values)

    return None


def _aggregate_top_values(df: pd.DataFrame, group_col: str, value_col: str, top_n: int) -> tuple[pd.DataFrame, str]:
    work = df[[group_col, value_col]].copy()
    positive_mask = _positive_boolean_mask(work[value_col])

    if positive_mask is not None:
        y_plot = f"{value_col}_count"
        work[y_plot] = positive_mask.astype(int)
        agg = (
            work.groupby(group_col, dropna=False)[y_plot]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        return agg, y_plot

    numeric_values = pd.to_numeric(work[value_col], errors="coerce")
    if numeric_values.notna().any():
        work[value_col] = numeric_values.fillna(0)
        agg = (
            work.groupby(group_col, dropna=False)[value_col]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        return agg, value_col

    agg = (
        work.groupby(group_col, dropna=False)
        .size()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index(name="count")
    )
    return agg, "count"


def build_visualization(session: SessionState, req: dict[str, Any]) -> dict[str, Any]:
    df = session.get_active_df()
    if df is None:
        raise ValueError("No active dataset to visualize.")

    chart_type = req.get("chart_type", "Bar Chart")
    template = req.get("template", "plotly_dark")
    colorscale = req.get("colorscale", "Blues")
    title = req.get("title") or chart_type

    color_val = req.get("color")
    color = None if not color_val or color_val == "None" else color_val

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    fig: Optional[go.Figure] = None

    if chart_type == "Bar Chart":
        x = req.get("x")
        y = req.get("y")
        top_n = req.get("top_n", 20)
        barmode = req.get("barmode", "group")
        if not x or not y:
            raise ValueError("Bar Chart requires both X and Y columns.")
        agg, y_plot = _aggregate_top_values(df, x, y, top_n)
        fig = px.bar(
            agg, x=x, y=y_plot,
            color=color if color and color in agg.columns else None,
            barmode=barmode,
            template=template,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )

    elif chart_type == "Line Chart":
        x = req.get("x")
        y_cols = req.get("y")
        if isinstance(y_cols, str):
            y_cols = [y_cols]
        if not x or not y_cols:
            raise ValueError("Line Chart requires an X axis and at least one Y metric.")
        melt = df[[x] + y_cols].melt(id_vars=[x], var_name="Series", value_name="Value")
        fig = px.line(melt, x=x, y="Value", color="Series", template=template)

    elif chart_type == "Scatter Plot":
        x = req.get("x")
        y = req.get("y")
        size_col = req.get("size")
        size = None if not size_col or size_col == "None" else size_col
        trendline = "ols" if req.get("trendline") else None
        if not x or not y:
            raise ValueError("Scatter Plot requires both X and Y columns.")
        fig = px.scatter(
            df, x=x, y=y, color=color, size=size,
            trendline=trendline, template=template, opacity=0.6,
        )

    elif chart_type == "Histogram":
        x = req.get("x")
        nbins = req.get("nbins", 30)
        if not x:
            raise ValueError("Histogram requires a column.")
        fig = px.histogram(
            df, x=x, nbins=nbins, color=color, template=template,
            marginal="box", color_discrete_sequence=["#4f8ef7"],
        )

    elif chart_type == "Box Plot":
        y = req.get("y")
        x = req.get("x")
        x_val = None if not x or x == "None" else x
        if not y:
            raise ValueError("Box Plot requires a value column.")
        fig = px.box(df, x=x_val, y=y, color=color, template=template, points="outliers")

    elif chart_type == "Violin Plot":
        y = req.get("y")
        x = req.get("x")
        x_val = None if not x or x == "None" else x
        if not y:
            raise ValueError("Violin Plot requires a value column.")
        fig = px.violin(df, x=x_val, y=y, color=color, template=template, box=True)

    elif chart_type == "Pie Chart":
        names = req.get("x") or req.get("names")
        values = req.get("y") or req.get("values")
        top_n = req.get("top_n", 8)
        hole = float(req.get("hole", 0.0))
        if not names or not values:
            raise ValueError("Pie Chart requires Labels and Values columns.")
        agg, values_plot = _aggregate_top_values(df, names, values, top_n)
        fig = px.pie(agg, names=names, values=values_plot, hole=hole, template=template)

    elif chart_type == "Area Chart":
        x = req.get("x")
        y = req.get("y")
        if not x or not y:
            raise ValueError("Area Chart requires both X and Y columns.")
        fig = px.area(df.sort_values(x), x=x, y=y, color=color, template=template)

    elif chart_type == "Heatmap":
        cols = req.get("hmap_cols") or numeric_cols[:8]
        if len(cols) < 2:
            raise ValueError("Heatmap requires at least 2 numeric columns.")
        corr = df[cols].corr()
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, template=template,
        )

    elif chart_type == "Treemap":
        path = req.get("path") or cat_cols[:2]
        values = req.get("values") or req.get("y")
        if not path or not values:
            raise ValueError("Treemap requires hierarchy categories and a values column.")
        fig = px.treemap(
            df, path=path, values=values, template=template,
            color=values, color_continuous_scale=colorscale,
        )

    elif chart_type == "Sunburst":
        path = req.get("path") or cat_cols[:2]
        values = req.get("values") or req.get("y")
        if not path or not values:
            raise ValueError("Sunburst requires hierarchy categories and a values column.")
        fig = px.sunburst(df, path=path, values=values, template=template)

    elif chart_type == "Bubble Chart":
        x = req.get("x")
        y = req.get("y")
        size = req.get("size")
        if not x or not y or not size:
            raise ValueError("Bubble Chart requires X, Y, and Size columns.")
        fig = px.scatter(
            df, x=x, y=y, size=size, color=color, template=template,
            size_max=60, opacity=0.7,
        )

    elif chart_type == "Funnel Chart":
        x = req.get("x")
        y = req.get("y")
        if not x or not y:
            raise ValueError("Funnel Chart requires Stage and Value columns.")
        agg, y_plot = _aggregate_top_values(df, x, y, len(df))
        fig = px.funnel(agg, x=y_plot, y=x, template=template)

    elif chart_type == "KPI Dashboard":
        kpi_cols = req.get("kpi_cols") or numeric_cols[:4]
        if not kpi_cols:
            raise ValueError("KPI Dashboard requires at least one numeric metric.")
        fig = go.Figure()
        for i, col in enumerate(kpi_cols):
            val = float(df[col].sum())
            avg = float(df[col].mean())
            fig.add_trace(go.Indicator(
                mode="number+delta",
                value=val,
                title={"text": col},
                delta={"reference": avg, "relative": True},
                domain={"row": 0, "column": i},
            ))
        fig.update_layout(
            grid={"rows": 1, "columns": len(kpi_cols)},
            template=template,
            height=200,
        )
    else:
        raise ValueError(f"Unknown chart type: {chart_type}")

    if fig is not None:
        fig.update_layout(title=title, margin=dict(l=20, r=20, t=50, b=20))
        fig_dict = json.loads(fig.to_json())
        return {
            "figure_spec": fig_dict,
            "chart_type": chart_type,
            "title": title,
        }
    raise ValueError("Failed to generate chart figure.")

from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


class VisualizationRequest(BaseModel):
    chart_type: str = Field(description="Bar Chart|Line Chart|Scatter Plot|Histogram|Box Plot|Violin Plot|Pie Chart|Area Chart|Heatmap|Treemap|Sunburst|Bubble Chart|Funnel Chart|KPI Dashboard")
    x: Optional[str] = None
    y: Optional[Any] = None  # str or list[str]
    color: Optional[str] = None
    size: Optional[str] = None
    barmode: Optional[str] = "group"
    top_n: Optional[int] = 20
    nbins: Optional[int] = 30
    hole: Optional[float] = 0.0
    path: Optional[list[str]] = None
    values: Optional[str] = None
    hmap_cols: Optional[list[str]] = None
    kpi_cols: Optional[list[str]] = None
    group: Optional[str] = None
    template: str = "plotly_dark"
    colorscale: str = "Blues"
    trendline: bool = False
    title: Optional[str] = None


class VisualizationResponse(BaseModel):
    figure_spec: dict[str, Any]
    chart_type: str
    title: str

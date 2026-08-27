from __future__ import annotations

import json
from typing import Any, Optional
import numpy as np
import pandas as pd

from backend.session.state import SessionState
from backend.services.dataset_service import sanitize_dataframe_for_json, to_json_compatible
from modules.insights import generate_statistical_insights, generate_llm_insights
from modules.forecasting import (
    moving_average_forecast,
    linear_trend_forecast,
    exponential_smoothing_forecast,
)
from modules.anomaly_detection import (
    detect_isolation_forest,
    detect_zscore,
    detect_iqr,
)


def get_quick_insights(session: SessionState) -> list[str]:
    df = session.get_active_df()
    if df is None:
        raise ValueError("No active dataset.")
    return generate_statistical_insights(df)


def get_ai_business_insights(session: SessionState, max_tokens: Optional[int] = 1500) -> dict[str, Any]:
    df = session.get_active_df()
    if df is None:
        raise ValueError("No active dataset.")

    api_key = session.config.get("hf_api_key", "")
    if not api_key:
        raise ValueError("Hugging Face API key is not configured. Please set your API key in Settings.")

    effective_tokens = max_tokens or session.config.get("max_tokens", 1500)
    data = generate_llm_insights(df, max_tokens=effective_tokens)
    session.last_insights = data
    return data


def run_forecast(
    session: SessionState,
    target_col: str,
    method: str = "Moving Average",
    horizon: int = 30,
    window: int = 7,
    alpha: float = 0.3,
) -> dict[str, Any]:
    df = session.get_active_df()
    if df is None:
        raise ValueError("No active dataset.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    series = df[target_col].dropna().reset_index(drop=True)
    if series.empty or not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"Column '{target_col}' has no numeric values to forecast.")

    if method == "Moving Average":
        result = moving_average_forecast(series, window=window, horizon=horizon)
    elif method == "Linear Trend":
        result = linear_trend_forecast(series, horizon=horizon)
    elif method == "Exponential Smoothing":
        result = exponential_smoothing_forecast(series, alpha=alpha, horizon=horizon)
    else:
        raise ValueError(f"Unsupported forecast method: {method}")

    fig_spec = None
    if result.fig is not None:
        try:
            fig_spec = json.loads(result.fig.to_json())
        except Exception:
            pass

    session.cached_forecast = result

    # Sanitize float lists for json
    def clean_floats(lst):
        return [None if np.isnan(v) or np.isinf(v) else float(v) for v in lst]

    return to_json_compatible({
        "method": result.method,
        "column": target_col,
        "horizon": int(result.horizon),
        "forecast_index": [int(x) if isinstance(x, (int, np.integer)) else str(x) for x in result.forecast_index],
        "forecast_values": clean_floats(result.forecast_values),
        "confidence_lower": clean_floats(result.confidence_lower),
        "confidence_upper": clean_floats(result.confidence_upper),
        "metrics": result.metrics or {},
        "interpretation": result.interpretation or "",
        "figure_spec": fig_spec,
        "historical_points": len(series),
    })


def run_anomaly_detection(
    session: SessionState,
    method: str = "Isolation Forest",
    contamination: float = 0.05,
    threshold: float = 3.0,
    factor: float = 1.5,
) -> dict[str, Any]:
    df = session.get_active_df()
    if df is None:
        raise ValueError("No active dataset.")

    numeric = df.select_dtypes(include=np.number)
    if numeric.empty:
        raise ValueError("Anomaly detection requires at least one numeric column.")

    if method == "Isolation Forest":
        result = detect_isolation_forest(df, contamination=contamination)
    elif method == "Z-Score":
        result = detect_zscore(df, threshold=threshold)
    elif method == "IQR":
        result = detect_iqr(df, factor=factor)
    else:
        raise ValueError(f"Unsupported anomaly method: {method}")

    fig_spec = None
    if result.fig is not None:
        try:
            fig_spec = json.loads(result.fig.to_json())
        except Exception:
            pass

    session.cached_anomaly = result

    anom_rows = []
    if isinstance(result.anomaly_df, pd.DataFrame) and not result.anomaly_df.empty:
        anom_rows = sanitize_dataframe_for_json(result.anomaly_df.head(100))

    anomaly_rate = round(result.n_anomalies / max(len(df), 1) * 100, 2)

    recs = [
        "Review flagged records to verify if anomalies represent genuine edge cases or data entry errors.",
        "Investigate root causes — check if anomalous rows cluster around specific categories or time periods.",
        "Consider cleaning or isolating verified erroneous records before downstream ML or reporting.",
    ]

    return to_json_compatible({
        "method": result.method,
        "n_anomalies": int(result.n_anomalies),
        "anomaly_rate": float(anomaly_rate),
        "anomaly_indices": [int(i) for i in result.anomaly_indices[:200]],
        "anomalous_rows": anom_rows,
        "scores": [float(s) for s in result.scores[:200]],
        "columns_used": result.columns_used,
        "figure_spec": fig_spec,
        "recommendations": recs,
    })

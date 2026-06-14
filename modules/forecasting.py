"""
Forecasting Module
Methods: Moving Average, Linear Trend, Exponential Smoothing, Seasonal Decomposition
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass, field


@dataclass
class ForecastResult:
    method: str = ""
    column: str = ""
    horizon: int = 0
    forecast_values: list = field(default_factory=list)
    forecast_index: list = field(default_factory=list)
    confidence_lower: list = field(default_factory=list)
    confidence_upper: list = field(default_factory=list)
    fig: object = None
    metrics: dict = field(default_factory=dict)
    interpretation: str = ""


def _make_time_index(df: pd.DataFrame, date_col: str = None) -> pd.Series:
    """Extract or create a time index."""
    if date_col and date_col in df.columns:
        return pd.to_datetime(df[date_col])
    # Try to find datetime column automatically
    for col in df.columns:
        try:
            s = pd.to_datetime(df[col], errors="coerce")
            if s.notna().sum() / len(df) > 0.8:
                return s
        except Exception:
            pass
    # Fall back to integer index
    return pd.RangeIndex(len(df))


def moving_average_forecast(
    series: pd.Series,
    window: int = 7,
    horizon: int = 30,
) -> ForecastResult:
    series = series.dropna()
    ma = series.rolling(window=window).mean()
    last_ma = ma.dropna().values[-1]
    std = series.rolling(window=window).std().dropna().values[-1]

    future_index = list(range(len(series), len(series) + horizon))
    forecast_vals = [last_ma] * horizon
    ci_lower = [last_ma - 1.96 * std] * horizon
    ci_upper = [last_ma + 1.96 * std] * horizon

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=series.values, name="Historical", line=dict(color="#4f8ef7")))
    fig.add_trace(go.Scatter(y=ma.values, name=f"MA({window})", line=dict(color="#f59e0b", dash="dot")))
    fig.add_trace(go.Scatter(
        x=future_index, y=forecast_vals, name="Forecast", line=dict(color="#10b981"),
    ))
    fig.add_trace(go.Scatter(
        x=future_index + future_index[::-1],
        y=ci_upper + ci_lower[::-1],
        fill="toself", fillcolor="rgba(16,185,129,0.1)",
        line=dict(color="rgba(255,255,255,0)"), name="95% CI",
    ))
    fig.update_layout(title=f"Moving Average Forecast (window={window})", template="plotly_dark")

    return ForecastResult(
        method="Moving Average",
        column=str(series.name),
        horizon=horizon,
        forecast_values=forecast_vals,
        forecast_index=future_index,
        confidence_lower=ci_lower,
        confidence_upper=ci_upper,
        fig=fig,
        interpretation=f"Based on a {window}-period moving average, the forecast for the next {horizon} periods is approximately {last_ma:.2f}.",
    )


def linear_trend_forecast(
    series: pd.Series,
    horizon: int = 30,
) -> ForecastResult:
    from sklearn.linear_model import LinearRegression

    series = series.dropna()
    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values

    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_

    future_X = np.arange(len(series), len(series) + horizon).reshape(-1, 1)
    forecast_vals = model.predict(future_X).tolist()

    residuals = y - model.predict(X)
    std_err = np.std(residuals)
    ci_lower = [v - 1.96 * std_err for v in forecast_vals]
    ci_upper = [v + 1.96 * std_err for v in forecast_vals]

    future_index = list(range(len(series), len(series) + horizon))
    trend_dir = "upward" if slope > 0 else "downward"

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=series.values, name="Historical", line=dict(color="#4f8ef7")))
    fig.add_trace(go.Scatter(y=model.predict(X).tolist(), name="Trend Line", line=dict(color="#f59e0b", dash="dot")))
    fig.add_trace(go.Scatter(x=future_index, y=forecast_vals, name="Forecast", line=dict(color="#10b981")))
    fig.add_trace(go.Scatter(
        x=future_index + future_index[::-1],
        y=ci_upper + ci_lower[::-1],
        fill="toself", fillcolor="rgba(16,185,129,0.1)",
        line=dict(color="rgba(255,255,255,0)"), name="95% CI",
    ))
    fig.update_layout(title="Linear Trend Forecast", template="plotly_dark")

    return ForecastResult(
        method="Linear Trend",
        column=str(series.name),
        horizon=horizon,
        forecast_values=forecast_vals,
        forecast_index=future_index,
        confidence_lower=ci_lower,
        confidence_upper=ci_upper,
        fig=fig,
        metrics={"slope": round(slope, 4), "intercept": round(float(intercept), 4), "r2": round(float(model.score(X, y)), 4)},
        interpretation=f"The data shows a {trend_dir} linear trend with slope {slope:.4f}. "
                       f"R² = {model.score(X, y):.3f}. Projected value in {horizon} periods: {forecast_vals[-1]:.2f}.",
    )


def exponential_smoothing_forecast(
    series: pd.Series,
    alpha: float = 0.3,
    horizon: int = 30,
) -> ForecastResult:
    series = series.dropna()

    # Simple exponential smoothing
    smoothed = series.ewm(alpha=alpha, adjust=False).mean()
    last_value = smoothed.iloc[-1]

    # Forecast: flat line at last smoothed value (SES property)
    future_index = list(range(len(series), len(series) + horizon))
    forecast_vals = [float(last_value)] * horizon

    residuals = series.values - smoothed.values
    std_err = np.std(residuals) * np.sqrt(np.arange(1, horizon + 1))
    ci_lower = [float(last_value) - 1.96 * s for s in std_err]
    ci_upper = [float(last_value) + 1.96 * s for s in std_err]

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=series.values, name="Historical", line=dict(color="#4f8ef7")))
    fig.add_trace(go.Scatter(y=smoothed.values, name=f"Smoothed (α={alpha})", line=dict(color="#f59e0b", dash="dot")))
    fig.add_trace(go.Scatter(x=future_index, y=forecast_vals, name="Forecast", line=dict(color="#10b981")))
    fig.add_trace(go.Scatter(
        x=future_index + future_index[::-1],
        y=ci_upper + ci_lower[::-1],
        fill="toself", fillcolor="rgba(16,185,129,0.1)",
        line=dict(color="rgba(255,255,255,0)"), name="95% CI (expanding)",
    ))
    fig.update_layout(title=f"Exponential Smoothing Forecast (α={alpha})", template="plotly_dark")

    return ForecastResult(
        method="Exponential Smoothing",
        column=str(series.name),
        horizon=horizon,
        forecast_values=forecast_vals,
        forecast_index=future_index,
        confidence_lower=ci_lower,
        confidence_upper=ci_upper,
        fig=fig,
        interpretation=f"Exponential smoothing (α={alpha}) gives a forecast of {last_value:.2f}. "
                       f"Higher α = more weight on recent values.",
    )

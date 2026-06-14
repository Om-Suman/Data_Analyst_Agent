"""Forecasting Page."""
import streamlit as st
import pandas as pd
import numpy as np
from modules.forecasting import (
    moving_average_forecast, linear_trend_forecast, exponential_smoothing_forecast
)
from utils.session import get_active_df


def render():
    st.title("🔮 Forecasting")
    df = get_active_df()
    if df is None:
        st.warning("No dataset loaded. Please upload data first.")
        return

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns available for forecasting.")
        return

    st.markdown("Project future values using statistical forecasting methods.")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### ⚙️ Settings")
        target_col = st.selectbox("Target Column", numeric_cols)
        method = st.selectbox("Forecasting Method", [
            "Moving Average", "Linear Trend", "Exponential Smoothing"
        ])
        horizon = st.slider("Forecast Horizon (periods)", 7, 365, 30)

        if method == "Moving Average":
            window = st.slider("Window Size", 2, 90, 7)
        elif method == "Exponential Smoothing":
            alpha = st.slider("Smoothing Factor (α)", 0.05, 0.95, 0.3, 0.05)
            st.caption("Higher α = more weight on recent values")

        run_btn = st.button("🚀 Run Forecast", type="primary", use_container_width=True)

    with col2:
        series = df[target_col].dropna().reset_index(drop=True)
        st.markdown(f"### 📊 {target_col} — {len(series):,} data points")

        # Show raw series
        import plotly.express as px
        fig_raw = px.line(y=series.values, template="plotly_dark",
                          title=f"Historical: {target_col}",
                          color_discrete_sequence=["#4f8ef7"])
        fig_raw.update_layout(xaxis_title="Period", yaxis_title=target_col)
        st.plotly_chart(fig_raw, use_container_width=True)

        if run_btn:
            with st.spinner("Running forecast..."):
                try:
                    if method == "Moving Average":
                        result = moving_average_forecast(series, window=window, horizon=horizon)
                    elif method == "Linear Trend":
                        result = linear_trend_forecast(series, horizon=horizon)
                    else:
                        result = exponential_smoothing_forecast(series, alpha=alpha, horizon=horizon)

                    st.session_state["forecast_results"] = result
                except Exception as e:
                    st.error(f"Forecasting error: {e}")
                    return

        if st.session_state.get("forecast_results"):
            result = st.session_state["forecast_results"]

            st.markdown(f"### 📈 Forecast Results — {result.method}")
            st.plotly_chart(result.fig, use_container_width=True)

            # Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Forecast Horizon", f"{result.horizon} periods")
            c2.metric("Final Forecast Value", f"{result.forecast_values[-1]:.2f}")
            if result.metrics:
                for key, val in list(result.metrics.items())[:1]:
                    c3.metric(key.upper(), f"{val:.4f}")

            # Interpretation
            if result.interpretation:
                st.info(f"💡 {result.interpretation}")

            # Export forecast
            forecast_df = pd.DataFrame({
                "Period": result.forecast_index,
                "Forecast": result.forecast_values,
                "Lower_CI": result.confidence_lower,
                "Upper_CI": result.confidence_upper,
            })
            csv = forecast_df.to_csv(index=False).encode()
            st.download_button("📥 Download Forecast", csv, "forecast.csv", "text/csv")

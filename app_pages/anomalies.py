"""Anomaly Detection Page."""
import streamlit as st
import pandas as pd
import numpy as np
from modules.anomaly_detection import detect_isolation_forest, detect_zscore, detect_iqr
from utils.session import get_active_df


def render():
    st.title("🚨 Anomaly Detection")
    df = get_active_df()
    if df is None:
        st.warning("No dataset loaded. Please upload data first.")
        return

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.warning("Need numeric columns for anomaly detection.")
        return

    st.markdown("Detect unusual patterns and outliers using multiple statistical methods.")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### ⚙️ Settings")
        method = st.selectbox("Detection Method", [
            "Isolation Forest", "Z-Score", "IQR"
        ])

        if method == "Isolation Forest":
            contamination = st.slider("Expected Anomaly Rate", 0.01, 0.2, 0.05, 0.01)
            st.caption("The expected proportion of anomalies in the dataset.")
        elif method == "Z-Score":
            threshold = st.slider("Z-Score Threshold", 1.5, 5.0, 3.0, 0.5)
            st.caption("Points beyond this many standard deviations are anomalies.")
        else:
            factor = st.slider("IQR Factor", 1.0, 3.0, 1.5, 0.25)
            st.caption("Points beyond Q1/Q3 ± factor×IQR are anomalies.")

        run_btn = st.button("🚀 Detect Anomalies", type="primary", use_container_width=True)

    with col2:
        if run_btn:
            with st.spinner("Running anomaly detection..."):
                try:
                    if method == "Isolation Forest":
                        result = detect_isolation_forest(df, contamination=contamination)
                    elif method == "Z-Score":
                        result = detect_zscore(df, threshold=threshold)
                    else:
                        result = detect_iqr(df, factor=factor)
                    st.session_state["anomaly_results"] = result
                except Exception as e:
                    st.error(f"Detection error: {e}")
                    return

        result = st.session_state.get("anomaly_results")
        if not result:
            st.info("Configure settings and click **Detect Anomalies** to begin.")
            return

        # KPIs
        pct = result.n_anomalies / len(df) * 100 if len(df) else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Method", result.method)
        c2.metric("Anomalies Found", f"{result.n_anomalies:,}")
        c3.metric("Anomaly Rate", f"{pct:.1f}%")

        # Severity badge
        if pct < 2:
            st.success("🟢 Low anomaly rate — data looks healthy.")
        elif pct < 10:
            st.warning("🟡 Moderate anomaly rate — review flagged records.")
        else:
            st.error("🔴 High anomaly rate — significant data quality issues may exist.")

        # Visualization
        if result.fig:
            st.plotly_chart(result.fig, use_container_width=True)

        # Anomalous records table
        if result.anomaly_df is not None and len(result.anomaly_df) > 0:
            st.markdown("### 🔎 Anomalous Records")
            st.dataframe(result.anomaly_df.head(50), use_container_width=True)

            csv = result.anomaly_df.to_csv(index=False).encode()
            st.download_button("📥 Download Anomalies", csv, "anomalies.csv", "text/csv")

            # Recommended actions
            st.markdown("### 💡 Recommended Actions")
            recs = [
                "**Review flagged records** — verify if anomalies are errors or valid edge cases.",
                "**Investigate root causes** — check if anomalies cluster around specific dates/categories.",
                "**Consider data cleaning** — remove confirmed erroneous records before analysis.",
                "**Document findings** — note intentional outliers for business context.",
            ]
            for rec in recs:
                st.markdown(f"- {rec}")

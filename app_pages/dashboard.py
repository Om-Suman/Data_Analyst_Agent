"""Dashboard — overview of loaded datasets and recent activity."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.session import get_active_df, get_active_meta
from datetime import datetime


def render():
    st.title("📊 Data Analyst Agent")

    datasets = st.session_state.get("datasets", {})
    query_history = st.session_state.get("query_history", [])

    if not datasets:
        _render_welcome()
        return

    df = get_active_df()
    meta = get_active_meta()
    active_name = st.session_state.active_dataset

    # --- KPI Row ---
    st.markdown("### 📈 Dataset Overview")
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("Rows", f"{meta.get('rows', 0):,}")
    with c2:
        st.metric("Columns", meta.get("cols", 0))
    with c3:
        missing = meta.get("missing_total", 0)
        total_cells = meta.get("rows", 1) * meta.get("cols", 1)
        st.metric("Missing Values", f"{missing:,}", delta=f"{missing/total_cells*100:.1f}%" if total_cells else "0%", delta_color="inverse")
    with c4:
        st.metric("Datasets Loaded", len(datasets))
    with c5:
        st.metric("Queries Run", len(query_history))

    st.markdown("---")

    # --- Two Column Layout ---
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### 🔍 Data Preview")
        if df is not None:
            st.dataframe(df.head(10), use_container_width=True, height=300)

    with col2:
        st.markdown("### 📊 Column Types")
        if meta:
            nc = len(meta.get("numeric_cols", []))
            cc = len(meta.get("categorical_cols", []))
            dc = len(meta.get("date_cols", []))
            other = meta.get("cols", 0) - nc - cc - dc
            labels = []
            values = []
            for label, val in [("Numeric", nc), ("Categorical", cc), ("Datetime", dc), ("Other", other)]:
                if val > 0:
                    labels.append(label)
                    values.append(val)
            fig = px.pie(
                values=values, names=labels,
                color_discrete_sequence=["#4f8ef7", "#10b981", "#f59e0b", "#8b5cf6"],
                template="plotly_dark", hole=0.5,
            )
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=250)
            st.plotly_chart(fig, use_container_width=True)

    # --- Quick Stats for numeric cols ---
    if df is not None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            st.markdown("### 📉 Distribution Snapshots")
            cols_to_show = numeric_cols[:4]
            cols = st.columns(len(cols_to_show))
            for i, col in enumerate(cols_to_show):
                with cols[i]:
                    fig = px.histogram(
                        df, x=col, nbins=30,
                        title=col,
                        template="plotly_dark",
                        color_discrete_sequence=["#4f8ef7"],
                    )
                    fig.update_layout(margin=dict(t=30, b=10, l=5, r=5), height=180, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

    # --- Recent Queries ---
    if query_history:
        st.markdown("### 🕒 Recent Queries")
        recent = query_history[-5:][::-1]
        for entry in recent:
            ts = entry["timestamp"].strftime("%H:%M:%S") if isinstance(entry["timestamp"], datetime) else str(entry["timestamp"])
            with st.expander(f"[{ts}] {entry['question'][:80]}", expanded=False):
                st.markdown(f"**Dataset:** {entry.get('dataset', 'N/A')}")
                if entry.get("code"):
                    with st.expander("Code", expanded=False):
                        st.code(entry["code"], language="python")
                if entry.get("result_summary"):
                    st.markdown(entry["result_summary"][:500])

    # --- All Datasets ---
    if len(datasets) > 1:
        st.markdown("### 📂 Loaded Datasets")
        rows = []
        for name, rec in datasets.items():
            rows.append({
                "Name": name,
                "Rows": f"{rec['rows']:,}",
                "Cols": rec["cols"],
                "Source": rec.get("source", "—"),
                "Loaded": rec["uploaded_at"].strftime("%H:%M") if hasattr(rec.get("uploaded_at"), "strftime") else "—",
                "Version": rec.get("version", 1),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _render_welcome():
    st.markdown("""
    <div style='text-align:center;padding:3rem 1rem'>
    <p style='color:#718096;font-size:1.1rem;max-width:600px;margin:1rem auto'>
    Upload your data and start exploring with AI-powered analytics, visualizations, and insights.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🚀 What can you do?")
    features = [
        ("📁", "Data Upload", "CSV, Excel, JSON, SQLite, PDF, Word, Images"),
        ("🧹", "Data Cleaning", "Auto-fix missing values, duplicates, outliers, types"),
        ("🤖", "AI Query", "Ask questions in plain English — get Pandas code + insights"),
        ("📈", "Visualizations", "Interactive Plotly charts with one click"),
        ("💡", "Auto Insights", "AI-generated business intelligence reports"),
        ("🔮", "Forecasting", "Moving averages, trend analysis, projections"),
        ("🚨", "Anomaly Detection", "Isolation Forest, Z-Score, IQR methods"),
        ("📄", "Reports", "Export to HTML, Excel, PDF"),
    ]
    cols = st.columns(4)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 4]:
            st.markdown(f"""
            <div class='metric-card' style='text-align:center'>
            <div style='font-size:2rem'>{icon}</div>
            <strong>{title}</strong><br>
            <small style='color:#718096'>{desc}</small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.info("👈 Go to **Data Upload** in the sidebar to get started.", icon="💡")

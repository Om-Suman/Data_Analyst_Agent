"""Reports Page — Export HTML, Excel, and PDF reports."""
import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime
from utils.session import get_active_df, get_active_meta


def render():
    st.title("📄 Reports")
    df = get_active_df()
    if df is None:
        st.warning("No dataset loaded. Please upload data first.")
        return

    meta = get_active_meta()
    name = st.session_state.active_dataset

    st.markdown("Export professional reports in multiple formats.")

    tab1, tab2, tab3 = st.tabs(["📊 HTML Report", "📈 Excel Report", "📋 Data Profile"])

    # --- HTML Report ---
    with tab1:
        st.markdown("### 📊 HTML Report")
        st.markdown("A standalone HTML file with dataset overview, statistics, and insights.")
        include_sample = st.checkbox("Include Data Sample (first 20 rows)", value=True)
        include_stats = st.checkbox("Include Statistics", value=True)
        include_insights = st.checkbox("Include Cached AI Insights", value=True)

        if st.button("📊 Generate HTML Report", type="primary"):
            with st.spinner("Building report..."):
                html = _build_html_report(df, meta, name, include_sample, include_stats, include_insights)
            st.download_button(
                "📥 Download HTML Report",
                data=html.encode(),
                file_name=f"{name}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
            )
            with st.expander("Preview (first 2000 chars)"):
                st.code(html[:2000], language="html")

    # --- Excel Report ---
    with tab2:
        st.markdown("### 📈 Excel Report")
        st.markdown("Multi-sheet Excel workbook with data, statistics, and query history.")

        if st.button("📊 Generate Excel Report", type="primary"):
            with st.spinner("Building Excel report..."):
                excel_bytes = _build_excel_report(df, meta, name)
            st.download_button(
                "📥 Download Excel Report",
                data=excel_bytes,
                file_name=f"{name}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            st.success("✅ Excel report ready!")

    # --- Data Profile ---
    with tab3:
        st.markdown("### 📋 Quick Data Profile")
        st.markdown("Fast in-app profiling (no external dependencies).")

        if st.button("Generate Profile", type="primary"):
            _render_profile(df, name)


def _build_html_report(df, meta, name, include_sample, include_stats, include_insights) -> str:
    from modules.insights import generate_statistical_insights
    static_insights = generate_statistical_insights(df)
    ai_insights = st.session_state.get("last_insights", {})

    numeric = df.select_dtypes(include=np.number)
    stats_html = numeric.describe().round(4).to_html(classes="table") if not numeric.empty else ""
    sample_html = df.head(20).to_html(classes="table", index=False) if include_sample else ""

    missing_html = df.isnull().sum().reset_index().rename(columns={"index": "Column", 0: "Missing"}).to_html(classes="table", index=False)

    insights_section = ""
    if include_insights and ai_insights:
        exec_sum = ai_insights.get("executive_summary", "")
        findings = "".join(f"<li>{f}</li>" for f in ai_insights.get("key_findings", []))
        recs = "".join(f"<li>{r}</li>" for r in ai_insights.get("recommendations", []))
        insights_section = f"""
        <h2>🤖 AI Insights</h2>
        <div class='card'><p>{exec_sum}</p></div>
        <h3>Key Findings</h3><ul>{findings}</ul>
        <h3>Recommendations</h3><ul>{recs}</ul>
        """

    static_items = "".join(f"<li>{i}</li>" for i in static_insights)

    return f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='UTF-8'>
<title>DataAgent Report: {name}</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #0e1117; color: #e2e8f0; margin: 0; padding: 2rem; }}
  h1 {{ color: #4f8ef7; border-bottom: 2px solid #4f8ef7; padding-bottom: 0.5rem; }}
  h2 {{ color: #10b981; margin-top: 2rem; }}
  h3 {{ color: #f59e0b; }}
  .card {{ background: #1a1d24; border: 1px solid #2d3748; border-radius: 8px; padding: 1rem; margin: 1rem 0; }}
  .kpi-row {{ display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }}
  .kpi {{ background: #1a2744; border: 1px solid #4f8ef7; border-radius: 8px; padding: 1rem 1.5rem; min-width: 150px; }}
  .kpi .value {{ font-size: 2rem; font-weight: 700; color: #4f8ef7; }}
  .kpi .label {{ color: #718096; font-size: 0.85rem; }}
  .table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.85rem; }}
  .table th {{ background: #1a2744; color: #4f8ef7; padding: 0.5rem; text-align: left; }}
  .table td {{ border: 1px solid #2d3748; padding: 0.4rem 0.6rem; }}
  .table tr:nth-child(even) {{ background: #161a21; }}
  footer {{ margin-top: 3rem; color: #718096; font-size: 0.8rem; border-top: 1px solid #2d3748; padding-top: 1rem; }}
</style>
</head>
<body>
<h1>📊 DataAgent Pro Report</h1>
<p style='color:#718096'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Dataset: <strong>{name}</strong></p>

<h2>📋 Dataset Overview</h2>
<div class='kpi-row'>
  <div class='kpi'><div class='value'>{len(df):,}</div><div class='label'>Total Rows</div></div>
  <div class='kpi'><div class='value'>{len(df.columns)}</div><div class='label'>Columns</div></div>
  <div class='kpi'><div class='value'>{df.isnull().sum().sum():,}</div><div class='label'>Missing Values</div></div>
  <div class='kpi'><div class='value'>{df.duplicated().sum():,}</div><div class='label'>Duplicate Rows</div></div>
  <div class='kpi'><div class='value'>{len(df.select_dtypes(include=np.number).columns)}</div><div class='label'>Numeric Cols</div></div>
</div>

<h2>🔍 Quick Insights</h2>
<ul>{static_items}</ul>

{"<h2>📊 Statistical Summary</h2><div class='card'>" + stats_html + "</div>" if include_stats else ""}

<h2>⚠️ Missing Values</h2>
<div class='card'>{missing_html}</div>

{"<h2>📋 Data Sample (First 20 Rows)</h2><div style='overflow-x:auto'>" + sample_html + "</div>" if include_sample else ""}

{insights_section}

<footer>Generated by DataAgent Pro | Powered by Hugging Face LLMs</footer>
</body>
</html>"""


def _build_excel_report(df, meta, name) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Sheet 1: Data
        df.to_excel(writer, sheet_name="Data", index=False)

        # Sheet 2: Statistics
        numeric = df.select_dtypes(include=np.number)
        if not numeric.empty:
            numeric.describe().round(4).to_excel(writer, sheet_name="Statistics")

        # Sheet 3: Missing Values
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ["Column", "Missing Count"]
        missing_df["Missing %"] = (missing_df["Missing Count"] / len(df) * 100).round(2)
        missing_df.to_excel(writer, sheet_name="Missing Values", index=False)

        # Sheet 4: Column Info
        col_info = []
        for col in df.columns:
            col_info.append({
                "Column": col,
                "Dtype": str(df[col].dtype),
                "Missing": df[col].isnull().sum(),
                "Unique": df[col].nunique(),
                "Sample": str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else "",
            })
        pd.DataFrame(col_info).to_excel(writer, sheet_name="Column Info", index=False)

        # Sheet 5: Query History
        history = st.session_state.get("query_history", [])
        if history:
            hist_df = pd.DataFrame([
                {"Timestamp": e["timestamp"], "Question": e["question"],
                 "Dataset": e.get("dataset", "")}
                for e in history
            ])
            hist_df.to_excel(writer, sheet_name="Query History", index=False)

    return output.getvalue()


def _render_profile(df, name):
    st.markdown(f"### 📋 Profile: {name}")

    numeric = df.select_dtypes(include=np.number)
    cat = df.select_dtypes(include="object")

    st.markdown("#### Numeric Columns")
    if not numeric.empty:
        stats = numeric.describe().T
        stats["skew"] = numeric.skew()
        stats["kurtosis"] = numeric.kurtosis()
        stats["missing"] = numeric.isnull().sum()
        stats["missing_pct"] = (numeric.isnull().sum() / len(df) * 100).round(1)
        st.dataframe(stats.round(3), use_container_width=True)

    st.markdown("#### Categorical Columns")
    if not cat.empty:
        cat_stats = []
        for col in cat.columns:
            cat_stats.append({
                "Column": col,
                "Unique Values": df[col].nunique(),
                "Most Common": str(df[col].mode()[0]) if df[col].notna().any() else "",
                "Most Common Count": int(df[col].value_counts().iloc[0]) if df[col].notna().any() else 0,
                "Missing": int(df[col].isnull().sum()),
                "Missing %": round(df[col].isnull().sum() / len(df) * 100, 1),
            })
        st.dataframe(pd.DataFrame(cat_stats), use_container_width=True)

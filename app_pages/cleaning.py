"""Data Cleaning Page."""
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from modules.cleaning import CleaningConfig, clean_dataframe, compute_quality_score
from utils.session import get_active_df, save_version


def render():
    st.title("🧹 Data Cleaning")
    df = get_active_df()
    if df is None:
        st.warning("No dataset loaded. Please upload data first.")
        return

    name = st.session_state.active_dataset
    score_before, grade_before = compute_quality_score(df)

    # Quality score banner
    color = "#10b981" if score_before >= 80 else "#f59e0b" if score_before >= 60 else "#ef4444"
    st.markdown(f"""
    <div style='background:#1a1d24;border:1px solid {color};border-radius:12px;padding:1rem;margin-bottom:1rem;display:flex;align-items:center;gap:1rem'>
    <div style='font-size:3rem'>{_grade_icon(grade_before)}</div>
    <div>
    <div style='font-size:1.5rem;font-weight:700;color:{color}'>Quality Score: {score_before}/100 (Grade {grade_before})</div>
    <div style='color:#718096'>Run cleaning to improve this score</div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Config UI ---
    st.markdown("### ⚙️ Cleaning Configuration")
    col1, col2 = st.columns(2)

    with col1:
        missing_strategy = st.selectbox(
            "Missing Value Strategy",
            ["mean", "median", "mode", "ffill", "bfill", "drop_rows", "drop_cols"],
            help="How to handle missing values",
        )
        custom_fill = None
        if missing_strategy == "custom":
            custom_fill = st.text_input("Custom Fill Value")

        remove_duplicates = st.checkbox("Remove Duplicate Rows", value=True)
        fix_dtypes = st.checkbox("Auto-Fix Data Types", value=True)
        normalize_cols = st.checkbox("Normalize Column Names (snake_case)", value=True)

    with col2:
        outlier_method = st.selectbox(
            "Outlier Detection Method",
            ["none", "zscore", "iqr"],
        )
        outlier_threshold = 3.0
        iqr_factor = 1.5
        if outlier_method == "zscore":
            outlier_threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.5)
        elif outlier_method == "iqr":
            iqr_factor = st.slider("IQR Factor", 1.0, 3.0, 1.5, 0.5)

        # Column selection
        all_cols = df.columns.tolist()
        selected_cols = st.multiselect(
            "Apply to specific columns (leave empty = all)",
            all_cols,
        )

    # --- Current Issues Summary ---
    st.markdown("### 🔎 Current Data Issues")
    issues_col1, issues_col2 = st.columns(2)

    with issues_col1:
        missing_per_col = df.isnull().sum()
        if missing_per_col.sum() > 0:
            st.markdown("**Missing Values by Column**")
            missing_df = missing_per_col[missing_per_col > 0].reset_index()
            missing_df.columns = ["Column", "Missing Count"]
            missing_df["Missing %"] = (missing_df["Missing Count"] / len(df) * 100).round(1)
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("✅ No missing values")

    with issues_col2:
        dup_count = df.duplicated().sum()
        st.metric("Duplicate Rows", f"{dup_count:,}", delta=f"{dup_count/len(df)*100:.1f}%" if dup_count else "0%", delta_color="inverse")
        st.markdown("")

        # Dtype overview
        dtype_df = df.dtypes.reset_index()
        dtype_df.columns = ["Column", "Type"]
        dtype_df["Type"] = dtype_df["Type"].astype(str)
        st.dataframe(dtype_df, use_container_width=True, height=200)

    # --- Run Cleaning ---
    if st.button("🚀 Run Cleaning", type="primary"):
        config = CleaningConfig(
            missing_strategy=missing_strategy,
            custom_fill_value=custom_fill,
            remove_duplicates=remove_duplicates,
            fix_dtypes=fix_dtypes,
            normalize_column_names=normalize_cols,
            outlier_method=outlier_method,
            outlier_threshold=outlier_threshold,
            iqr_factor=iqr_factor,
            columns_to_clean=selected_cols if selected_cols else None,
        )

        with st.spinner("Cleaning data..."):
            cleaned_df, report = clean_dataframe(df, config)

        # Show report
        st.markdown("---")
        st.markdown("### ✅ Cleaning Report")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Rows Before", f"{report.rows_before:,}")
        r2.metric("Rows After", f"{report.rows_after:,}", delta=f"-{report.rows_before - report.rows_after:,}")
        r3.metric("Duplicates Removed", f"{report.duplicates_removed:,}")
        r4.metric("Outliers Removed", f"{report.outliers_removed:,}")

        score_after = report.quality_score_after
        grade_after = report.quality_grade_after
        delta_score = score_after - score_before
        st.metric("Quality Score", f"{score_after}/100 ({grade_after})",
                  delta=f"+{delta_score:.1f}" if delta_score >= 0 else f"{delta_score:.1f}")

        if report.missing_filled:
            filled_count = sum(v for v in report.missing_filled.values() if isinstance(v, int))
            st.info(f"Filled {filled_count:,} missing values using strategy: **{missing_strategy}**")

        if report.dtype_changes:
            with st.expander(f"🔄 {len(report.dtype_changes)} dtype changes"):
                for col, change in report.dtype_changes.items():
                    st.write(f"- **{col}**: {change}")

        if report.col_renames:
            with st.expander(f"📝 {len(report.col_renames)} column renames"):
                for old, new in report.col_renames.items():
                    st.write(f"- `{old}` → `{new}`")

        if report.recommendations:
            st.markdown("**💡 Recommendations:**")
            for rec in report.recommendations:
                st.write(f"- {rec}")

        st.markdown("### Download Cleaned Dataset")
        csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
        excel_bytes = _to_excel_bytes(cleaned_df)
        safe_name = _safe_download_name(name)

        download_col1, download_col2, _ = st.columns([1, 1, 4])
        download_col1.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name=f"{safe_name}_cleaned.csv",
            mime="text/csv",
            use_container_width=True,
        )
        download_col2.download_button(
            "Download Excel",
            data=excel_bytes,
            file_name=f"{safe_name}_cleaned.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        # Apply to session
        col_apply, col_discard = st.columns([1, 5])
        if col_apply.button("✅ Apply Cleaning", type="primary"):
            save_version(name, "Before cleaning")
            st.session_state.datasets[name]["df"] = cleaned_df
            st.session_state.datasets[name]["rows"] = len(cleaned_df)
            st.session_state.datasets[name]["cols"] = len(cleaned_df.columns)
            st.session_state.cleaning_log.append({
                "dataset": name,
                "config": str(config),
                "report": report,
            })
            st.success("✅ Cleaned dataset applied!")
            st.rerun()

def _to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Cleaned Data")
    return buffer.getvalue()


def _safe_download_name(name: str) -> str:
    base_name = str(name).rsplit(".", 1)[0]
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in base_name)
    return safe_name or "dataset"


def _grade_icon(grade):
    return {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🟠", "F": "🔴"}.get(grade, "⚪")

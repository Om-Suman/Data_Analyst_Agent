"""Data Explorer Page."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.session import get_active_df, get_active_meta


def render():
    st.title("🔍 Data Explorer")
    df = get_active_df()
    if df is None:
        st.warning("No dataset loaded. Please upload data first.")
        return

    meta = get_active_meta()
    name = st.session_state.active_dataset

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Browse", "📊 Distributions", "🔗 Correlations", "🔎 Filter & Search", "📐 Column Profiler"
    ])

    # --- TAB 1: Browse ---
    with tab1:
        st.markdown(f"### {name} — {df.shape[0]:,} rows × {df.shape[1]} columns")
        cols_to_show = st.multiselect("Show columns", df.columns.tolist(), default=df.columns.tolist()[:10])
        n_rows = st.slider("Rows to display", 10, min(500, len(df)), 50)
        if cols_to_show:
            st.dataframe(df[cols_to_show].head(n_rows), use_container_width=True)
        else:
            st.dataframe(df.head(n_rows), use_container_width=True)

        # Download
        csv = df.to_csv(index=False).encode()
        st.download_button("📥 Download CSV", csv, file_name=f"{name}_export.csv", mime="text/csv")

    # --- TAB 2: Distributions ---
    with tab2:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()

        if numeric_cols:
            st.markdown("#### Numeric Distributions")
            col_select = st.selectbox("Select column", numeric_cols, key="dist_col")
            chart_type = st.radio("Chart type", ["Histogram", "Box Plot", "Violin"], horizontal=True, key="dist_type")

            if chart_type == "Histogram":
                nbins = st.slider("Bins", 10, 100, 30)
                fig = px.histogram(df, x=col_select, nbins=nbins, template="plotly_dark",
                                   color_discrete_sequence=["#4f8ef7"], marginal="box")
            elif chart_type == "Box Plot":
                group_col = st.selectbox("Group by (optional)", ["None"] + cat_cols, key="box_group")
                color = group_col if group_col != "None" else None
                fig = px.box(df, y=col_select, color=color, template="plotly_dark", points="outliers")
            else:
                group_col = st.selectbox("Group by (optional)", ["None"] + cat_cols, key="violin_group")
                color = group_col if group_col != "None" else None
                fig = px.violin(df, y=col_select, color=color, template="plotly_dark", box=True)

            st.plotly_chart(fig, use_container_width=True)

        if cat_cols:
            st.markdown("#### Categorical Distributions")
            cat_col = st.selectbox("Select column", cat_cols, key="cat_col")
            top_n = st.slider("Top N values", 5, 30, 15)
            vc = df[cat_col].value_counts().head(top_n).reset_index()
            vc.columns = [cat_col, "Count"]
            fig = px.bar(vc, x=cat_col, y="Count", template="plotly_dark",
                         color="Count", color_continuous_scale="Blues", title=f"Top {top_n}: {cat_col}")
            st.plotly_chart(fig, use_container_width=True)

    # --- TAB 3: Correlations ---
    with tab3:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) < 2:
            st.info("Need at least 2 numeric columns for correlation analysis.")
        else:
            cols_for_corr = st.multiselect("Select columns", numeric_cols, default=numeric_cols[:10])
            if len(cols_for_corr) >= 2:
                method = st.radio("Method", ["pearson", "spearman", "kendall"], horizontal=True)
                corr_matrix = df[cols_for_corr].corr(method=method)

                fig = px.imshow(
                    corr_matrix, text_auto=".2f", template="plotly_dark",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title=f"{method.title()} Correlation Matrix",
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

                # Top correlations
                st.markdown("#### 🔝 Strongest Correlations")
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_pairs.append({
                            "Col A": corr_matrix.columns[i],
                            "Col B": corr_matrix.columns[j],
                            "Correlation": round(corr_matrix.iloc[i, j], 4),
                            "Strength": abs(corr_matrix.iloc[i, j])
                        })
                corr_df = pd.DataFrame(corr_pairs).sort_values("Strength", ascending=False).drop("Strength", axis=1)
                st.dataframe(corr_df.head(15), use_container_width=True)

                # Scatter for top pair
                if corr_pairs:
                    top_pair = sorted(corr_pairs, key=lambda x: abs(x["Correlation"]), reverse=True)[0]
                    st.markdown(f"#### Scatter: {top_pair['Col A']} vs {top_pair['Col B']}")
                    color_option = st.selectbox("Color by", ["None"] + df.select_dtypes(include="object").columns.tolist(), key="scatter_color")
                    fig2 = px.scatter(
                        df, x=top_pair["Col A"], y=top_pair["Col B"],
                        color=None if color_option == "None" else color_option,
                        trendline="ols", template="plotly_dark", opacity=0.6,
                    )
                    st.plotly_chart(fig2, use_container_width=True)

    # --- TAB 4: Filter & Search ---
    with tab4:
        st.markdown("### 🔎 Filter Data")
        filtered_df = df.copy()
        filters_applied = []

        with st.expander("Add Filters", expanded=True):
            filter_col = st.selectbox("Column", df.columns.tolist())
            col_dtype = df[filter_col].dtype

            if pd.api.types.is_numeric_dtype(col_dtype):
                min_val, max_val = float(df[filter_col].min()), float(df[filter_col].max())
                range_val = st.slider(f"{filter_col} range", min_val, max_val, (min_val, max_val))
                filtered_df = filtered_df[
                    (filtered_df[filter_col] >= range_val[0]) &
                    (filtered_df[filter_col] <= range_val[1])
                ]
                filters_applied.append(f"{filter_col} ∈ [{range_val[0]:.2f}, {range_val[1]:.2f}]")
            else:
                unique_vals = df[filter_col].dropna().unique().tolist()
                selected_vals = st.multiselect(f"{filter_col} values", unique_vals, default=unique_vals[:5])
                if selected_vals:
                    filtered_df = filtered_df[filtered_df[filter_col].isin(selected_vals)]
                    filters_applied.append(f"{filter_col} in {selected_vals}")

        if filters_applied:
            st.info(f"Active filters: {' | '.join(filters_applied)} → **{len(filtered_df):,}** rows")

        st.dataframe(filtered_df, use_container_width=True)

        if len(filtered_df) < len(df):
            csv_filtered = filtered_df.to_csv(index=False).encode()
            st.download_button("📥 Download Filtered Data", csv_filtered,
                               file_name="filtered_data.csv", mime="text/csv")

    # --- TAB 5: Column Profiler ---
    with tab5:
        st.markdown("### 📐 Column Profiler")
        profile_col = st.selectbox("Select column to profile", df.columns.tolist())
        col_data = df[profile_col]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", f"{len(col_data):,}")
        c2.metric("Missing", f"{col_data.isnull().sum():,}")
        c3.metric("Unique", f"{col_data.nunique():,}")
        c4.metric("Type", str(col_data.dtype))

        if pd.api.types.is_numeric_dtype(col_data):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Min", f"{col_data.min():.4g}")
            c2.metric("Max", f"{col_data.max():.4g}")
            c3.metric("Mean", f"{col_data.mean():.4g}")
            c4.metric("Std Dev", f"{col_data.std():.4g}")

            s1, s2, s3 = st.columns(3)
            s1.metric("Skewness", f"{col_data.skew():.3f}")
            s2.metric("Kurtosis", f"{col_data.kurtosis():.3f}")
            s3.metric("Zeros", f"{(col_data == 0).sum():,}")

            fig = px.histogram(df, x=profile_col, nbins=40, template="plotly_dark",
                               color_discrete_sequence=["#4f8ef7"], marginal="rug")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("**Top Values:**")
            vc = col_data.value_counts().head(20).reset_index()
            vc.columns = ["Value", "Count"]
            vc["Percent"] = (vc["Count"] / len(df) * 100).round(1)
            st.dataframe(vc, use_container_width=True)
            fig = px.bar(vc, x="Value", y="Count", template="plotly_dark",
                         color_discrete_sequence=["#4f8ef7"])
            st.plotly_chart(fig, use_container_width=True)

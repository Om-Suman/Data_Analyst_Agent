"""Visualization Studio Page."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.session import get_active_df


CHART_TYPES = [
    "Bar Chart", "Line Chart", "Scatter Plot", "Histogram",
    "Box Plot", "Violin Plot", "Pie Chart", "Area Chart",
    "Heatmap", "Treemap", "Sunburst", "Bubble Chart",
    "Funnel Chart", "KPI Dashboard",
]


def render():
    st.title("📈 Visualization Studio")

    df = get_active_df()
    if df is None:
        st.warning("No dataset loaded. Please upload data first.")
        return

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    all_cols = df.columns.tolist()

    col_left, col_right = st.columns([1, 3])

    with col_left:
        st.markdown("### ⚙️ Chart Builder")
        chart_type = st.selectbox("Chart Type", CHART_TYPES)

        st.markdown("---")
        _render_controls(df, chart_type, numeric_cols, cat_cols, date_cols, all_cols)

    with col_right:
        _render_chart(df, chart_type, numeric_cols, cat_cols, date_cols, all_cols)


def _render_controls(df, chart_type, numeric_cols, cat_cols, date_cols, all_cols):
    """Render chart-specific controls into session state."""
    # Color theme
    template = st.selectbox("Theme", ["plotly_dark", "plotly", "plotly_white", "ggplot2", "seaborn"])
    st.session_state["viz_template"] = template

    color_scale = st.selectbox("Color Scale",
        ["Blues", "Reds", "Greens", "Viridis", "Plasma", "RdBu", "Spectral", "Turbo"])
    st.session_state["viz_colorscale"] = color_scale

    st.markdown("---")
    chart_key = chart_type.lower().replace(" ", "_")

    if chart_type == "Bar Chart":
        st.session_state["viz_x"] = st.selectbox("X Axis", all_cols, key="bar_x")
        st.session_state["viz_y"] = st.selectbox("Y Axis", all_cols, key="bar_y")
        st.session_state["viz_color"] = st.selectbox("Color by", ["None"] + cat_cols, key="bar_color")
        st.session_state["viz_barmode"] = st.radio("Bar Mode", ["group", "stack", "overlay"], horizontal=True)
        st.session_state["viz_top_n"] = st.slider("Top N", 5, 50, 20)

    elif chart_type == "Line Chart":
        x_options = date_cols + numeric_cols + cat_cols
        st.session_state["viz_x"] = st.selectbox("X Axis", x_options, key="line_x")
        st.session_state["viz_y"] = st.multiselect("Y Axis (multi)", numeric_cols, default=numeric_cols[:1], key="line_y")
        st.session_state["viz_color"] = st.selectbox("Color by", ["None"] + cat_cols, key="line_color")

    elif chart_type == "Scatter Plot":
        st.session_state["viz_x"] = st.selectbox("X Axis", numeric_cols, key="scatter_x")
        st.session_state["viz_y"] = st.selectbox("Y Axis", numeric_cols, index=min(1, len(numeric_cols)-1), key="scatter_y")
        st.session_state["viz_color"] = st.selectbox("Color by", ["None"] + cat_cols + numeric_cols, key="scatter_color")
        st.session_state["viz_size"] = st.selectbox("Size by", ["None"] + numeric_cols, key="scatter_size")
        st.session_state["viz_trendline"] = st.checkbox("Add Trendline")

    elif chart_type == "Histogram":
        st.session_state["viz_x"] = st.selectbox("Column", numeric_cols, key="hist_x")
        st.session_state["viz_nbins"] = st.slider("Bins", 10, 100, 30)
        st.session_state["viz_color"] = st.selectbox("Color by", ["None"] + cat_cols, key="hist_color")

    elif chart_type in ("Box Plot", "Violin Plot"):
        st.session_state["viz_y"] = st.selectbox("Value Column", numeric_cols, key="box_y")
        st.session_state["viz_x"] = st.selectbox("Group by", ["None"] + cat_cols, key="box_x")
        st.session_state["viz_color"] = st.selectbox("Color by", ["None"] + cat_cols, key="box_color")

    elif chart_type == "Pie Chart":
        st.session_state["viz_names"] = st.selectbox("Labels", cat_cols, key="pie_names")
        st.session_state["viz_values"] = st.selectbox("Values", numeric_cols, key="pie_values")
        st.session_state["viz_top_n"] = st.slider("Top N slices", 3, 20, 8)
        st.session_state["viz_hole"] = st.slider("Donut Hole", 0.0, 0.8, 0.0, 0.1)

    elif chart_type == "Area Chart":
        x_opts = date_cols + numeric_cols
        st.session_state["viz_x"] = st.selectbox("X Axis", x_opts, key="area_x")
        st.session_state["viz_y"] = st.selectbox("Y Axis", numeric_cols, key="area_y")
        st.session_state["viz_color"] = st.selectbox("Color by", ["None"] + cat_cols, key="area_color")

    elif chart_type == "Heatmap":
        st.session_state["viz_hmap_cols"] = st.multiselect("Columns", numeric_cols, default=numeric_cols[:8])

    elif chart_type == "Treemap":
        st.session_state["viz_path"] = st.multiselect("Hierarchy (in order)", cat_cols, default=cat_cols[:2])
        st.session_state["viz_values"] = st.selectbox("Values", numeric_cols, key="tree_vals")

    elif chart_type == "Sunburst":
        st.session_state["viz_path"] = st.multiselect("Hierarchy", cat_cols, default=cat_cols[:2], key="sun_path")
        st.session_state["viz_values"] = st.selectbox("Values", numeric_cols, key="sun_vals")

    elif chart_type == "Bubble Chart":
        st.session_state["viz_x"] = st.selectbox("X", numeric_cols, key="bub_x")
        st.session_state["viz_y"] = st.selectbox("Y", numeric_cols, index=min(1, len(numeric_cols)-1), key="bub_y")
        st.session_state["viz_size"] = st.selectbox("Bubble Size", numeric_cols, index=min(2, len(numeric_cols)-1), key="bub_size")
        st.session_state["viz_color"] = st.selectbox("Color", ["None"] + cat_cols, key="bub_color")

    elif chart_type == "Funnel Chart":
        st.session_state["viz_x"] = st.selectbox("Stage (category)", cat_cols, key="fun_x")
        st.session_state["viz_y"] = st.selectbox("Value", numeric_cols, key="fun_y")

    elif chart_type == "KPI Dashboard":
        st.session_state["viz_kpi_cols"] = st.multiselect("KPI Metrics", numeric_cols, default=numeric_cols[:4])
        st.session_state["viz_group"] = st.selectbox("Group by", ["None"] + cat_cols, key="kpi_group")


def _render_chart(df, chart_type, numeric_cols, cat_cols, date_cols, all_cols):
    template = st.session_state.get("viz_template", "plotly_dark")
    colorscale = st.session_state.get("viz_colorscale", "Blues")
    title = st.text_input("Chart Title", value=chart_type)

    try:
        fig = _build_fig(df, chart_type, template, colorscale, numeric_cols, cat_cols)
        if fig:
            if title:
                fig.update_layout(title=title)
            st.plotly_chart(fig, use_container_width=True)

            # Save to history
            if st.button("💾 Save to Viz History"):
                if "viz_history" not in st.session_state:
                    st.session_state.viz_history = []
                st.session_state.viz_history.append({"title": title, "type": chart_type})
                st.success("Saved!")

    except Exception as e:
        st.error(f"Chart error: {e}")


def _build_fig(df, chart_type, template, colorscale, numeric_cols, cat_cols):
    ss = st.session_state
    color_val = ss.get("viz_color")
    color = None if color_val == "None" or not color_val else color_val

    if chart_type == "Bar Chart":
        x = ss.get("viz_x")
        y = ss.get("viz_y")
        top_n = ss.get("viz_top_n", 20)
        barmode = ss.get("viz_barmode", "group")
        if not x or not y:
            return None
        agg = df.groupby(x)[y].sum().nlargest(top_n).reset_index()
        return px.bar(agg, x=x, y=y, color=color if color and color in agg.columns else None,
                      barmode=barmode, template=template, color_discrete_sequence=px.colors.qualitative.Set2)

    elif chart_type == "Line Chart":
        x = ss.get("viz_x")
        y_cols = ss.get("viz_y", [])
        if not x or not y_cols:
            return None
        melt = df[[x] + y_cols].melt(id_vars=[x], var_name="Series", value_name="Value")
        return px.line(melt, x=x, y="Value", color="Series", template=template)

    elif chart_type == "Scatter Plot":
        x = ss.get("viz_x")
        y = ss.get("viz_y")
        size_col = ss.get("viz_size")
        size = None if size_col == "None" or not size_col else size_col
        trendline = "ols" if ss.get("viz_trendline") else None
        if not x or not y:
            return None
        return px.scatter(df, x=x, y=y, color=color, size=size,
                          trendline=trendline, template=template, opacity=0.6)

    elif chart_type == "Histogram":
        x = ss.get("viz_x")
        nbins = ss.get("viz_nbins", 30)
        if not x:
            return None
        return px.histogram(df, x=x, nbins=nbins, color=color, template=template,
                            marginal="box", color_discrete_sequence=["#4f8ef7"])

    elif chart_type == "Box Plot":
        y = ss.get("viz_y")
        x = ss.get("viz_x")
        x_val = None if x == "None" or not x else x
        return px.box(df, x=x_val, y=y, color=color, template=template, points="outliers")

    elif chart_type == "Violin Plot":
        y = ss.get("viz_y")
        x = ss.get("viz_x")
        x_val = None if x == "None" or not x else x
        return px.violin(df, x=x_val, y=y, color=color, template=template, box=True)

    elif chart_type == "Pie Chart":
        names = ss.get("viz_names")
        values = ss.get("viz_values")
        top_n = ss.get("viz_top_n", 8)
        hole = ss.get("viz_hole", 0.0)
        if not names or not values:
            return None
        agg = df.groupby(names)[values].sum().nlargest(top_n).reset_index()
        return px.pie(agg, names=names, values=values, hole=hole, template=template)

    elif chart_type == "Area Chart":
        x = ss.get("viz_x")
        y = ss.get("viz_y")
        if not x or not y:
            return None
        return px.area(df.sort_values(x), x=x, y=y, color=color, template=template)

    elif chart_type == "Heatmap":
        cols = ss.get("viz_hmap_cols", numeric_cols[:8])
        if len(cols) < 2:
            return None
        corr = df[cols].corr()
        return px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, template=template)

    elif chart_type == "Treemap":
        path = ss.get("viz_path", [])
        values = ss.get("viz_values")
        if not path or not values:
            return None
        return px.treemap(df, path=path, values=values, template=template,
                          color=values, color_continuous_scale=colorscale)

    elif chart_type == "Sunburst":
        path = ss.get("viz_path", [])
        values = ss.get("viz_values")
        if not path or not values:
            return None
        return px.sunburst(df, path=path, values=values, template=template)

    elif chart_type == "Bubble Chart":
        x = ss.get("viz_x")
        y = ss.get("viz_y")
        size = ss.get("viz_size")
        if not x or not y or not size:
            return None
        return px.scatter(df, x=x, y=y, size=size, color=color, template=template,
                          size_max=60, opacity=0.7)

    elif chart_type == "Funnel Chart":
        x = ss.get("viz_x")
        y = ss.get("viz_y")
        if not x or not y:
            return None
        agg = df.groupby(x)[y].sum().reset_index().sort_values(y, ascending=False)
        return px.funnel(agg, x=y, y=x, template=template)

    elif chart_type == "KPI Dashboard":
        kpi_cols = ss.get("viz_kpi_cols", numeric_cols[:4])
        group_col = ss.get("viz_group")
        if not kpi_cols:
            return None

        fig = go.Figure()
        for i, col in enumerate(kpi_cols):
            val = df[col].sum()
            avg = df[col].mean()
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
        return fig

    return None

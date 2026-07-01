"""Data Upload Page."""
import streamlit as st
import pandas as pd
from modules.ingestion import parse_uploaded_file
from modules.document_rag import build_document_bundle, store_document_bundle
from utils.session import register_dataset, register_text_dataset


def render():
    st.title("📁 Data Upload")
    st.markdown("Upload one or more datasets to begin analysis.")

    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["csv", "xlsx", "xls", "json", "db", "txt", "docx", "pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Supported: CSV, Excel, JSON, SQLite, Text, Word, PDF, Images (OCR)",
    )

    if not uploaded_files:
        _render_sample_loader()
        return

    for uploaded_file in uploaded_files:
        with st.expander(f"📂 {uploaded_file.name}", expanded=True):
            file_size_kb = uploaded_file.size / 1024
            st.caption(f"Size: {file_size_kb:.1f} KB")

            # Check already loaded
            if uploaded_file.name in st.session_state.datasets:
                st.info(f"Already loaded: {uploaded_file.name}")
                if st.button(f"Reload {uploaded_file.name}", key=f"reload_{uploaded_file.name}"):
                    _process_file(uploaded_file)
            else:
                _process_file(uploaded_file)

    # Show all loaded datasets
    if st.session_state.datasets:
        st.markdown("---")
        st.markdown("### 📂 Loaded Datasets")
        for name, rec in st.session_state.datasets.items():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            col1.markdown(f"**{name}**")
            col2.write(f"{rec['rows']:,} rows")
            col3.write(f"{rec['cols']} cols")
            if col4.button("Set Active", key=f"activate_{name}"):
                st.session_state.active_dataset = name
                st.rerun()


def _process_file(uploaded_file):
    with st.spinner(f"Processing {uploaded_file.name}..."):
        try:
            df, meta, data_type = parse_uploaded_file(uploaded_file)
        except Exception as e:
            st.error(f"Failed to parse file: {e}")
            return

    if data_type == "dataframe" and df is not None:
        register_dataset(uploaded_file.name, df, source=uploaded_file.name, meta=meta)
        st.success(f"✅ Loaded **{uploaded_file.name}**: {df.shape[0]:,} rows × {df.shape[1]} columns")

        # Preview
        tab1, tab2, tab3 = st.tabs(["Preview", "Column Info", "Statistics"])
        with tab1:
            st.dataframe(df.head(20), use_container_width=True)
        with tab2:
            col_info = []
            for col in df.columns:
                col_info.append({
                    "Column": col,
                    "Type": str(df[col].dtype),
                    "Missing": int(df[col].isnull().sum()),
                    "Missing %": f"{df[col].isnull().sum()/len(df)*100:.1f}%",
                    "Unique": int(df[col].nunique()),
                })
            st.dataframe(pd.DataFrame(col_info), use_container_width=True)
        with tab3:
            numeric = df.select_dtypes(include="number")
            if not numeric.empty:
                st.dataframe(numeric.describe().round(4), use_container_width=True)

    elif data_type == "text":
        content = meta.get("content", "")
        st.info(f"📄 Text document loaded: {len(content):,} characters")
        st.text_area("Content Preview", content[:2000], height=200)
        register_text_dataset(uploaded_file.name, content, source=uploaded_file.name, meta=meta)
        bundle = build_document_bundle(content, metadata={"name": uploaded_file.name, **meta})
        store_document_bundle(uploaded_file.name, bundle)
        st.success(f"✅ Indexed text document: {uploaded_file.name}")


def _render_sample_loader():
    st.markdown("---")
    st.markdown("### 🧪 Or load a sample dataset")
    samples = {
        "Sales Data": _make_sales_sample,
        "Employee Data": _make_employee_sample,
        "Finance Data": _make_finance_sample,
    }
    cols = st.columns(len(samples))
    for i, (label, fn) in enumerate(samples.items()):
        if cols[i].button(f"Load {label}", use_container_width=True):
            df = fn()
            register_dataset(label, df, source="sample")
            st.success(f"✅ Loaded sample: {label}")
            st.rerun()


def _make_sales_sample():
    import numpy as np
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "region": np.random.choice(["North", "South", "East", "West"], n),
        "product": np.random.choice(["Widget A", "Widget B", "Widget C", "Gadget X"], n),
        "sales": np.random.normal(1000, 300, n).clip(50).round(2),
        "units": np.random.randint(1, 100, n),
        "profit": np.random.normal(200, 80, n).round(2),
        "customer_id": np.random.randint(1000, 9999, n),
    })


def _make_employee_sample():
    import numpy as np
    np.random.seed(7)
    n = 300
    return pd.DataFrame({
        "employee_id": range(1, n+1),
        "department": np.random.choice(["Engineering", "Sales", "Marketing", "HR", "Finance"], n),
        "salary": np.random.normal(75000, 20000, n).clip(30000).round(0),
        "years_experience": np.random.randint(0, 20, n),
        "performance_score": np.random.uniform(1, 5, n).round(1),
        "remote": np.random.choice([True, False], n),
        "hire_date": pd.date_range("2015-01-01", periods=n, freq="30D"),
    })


def _make_finance_sample():
    import numpy as np
    np.random.seed(99)
    n = 365
    base = 100
    returns = np.random.normal(0.001, 0.02, n)
    prices = base * (1 + returns).cumprod()
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n),
        "close": prices.round(2),
        "volume": np.random.randint(1_000_000, 10_000_000, n),
        "high": (prices * np.random.uniform(1.0, 1.03, n)).round(2),
        "low": (prices * np.random.uniform(0.97, 1.0, n)).round(2),
        "category": np.random.choice(["Tech", "Finance", "Healthcare", "Energy"], n),
    })

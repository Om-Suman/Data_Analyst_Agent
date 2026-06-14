"""Sidebar navigation and page routing."""

import streamlit as st

PAGES = {
    "📊 Dashboard": "dashboard",
    "📁 Data Upload": "upload",
    "🧹 Data Cleaning": "cleaning",
    "🔍 Data Explorer": "explorer",
    "🤖 AI Query": "ai_query",
    "📈 Visualizations": "visualizations",
    "💡 Insights": "insights",
    "🔮 Forecasting": "forecasting",
    "🚨 Anomaly Detection": "anomalies",
    "📄 Reports": "reports",
    "⚙️ Settings": "settings",
}


def render_navigation():
    # Ensure active_page exists
    if "active_page" not in st.session_state:
        st.session_state.active_page = "dashboard"

    with st.sidebar:
        st.markdown("## 📊 DataAnalystAgent")
        st.markdown("---")

        # API Key Input
        if not st.session_state.get("hf_api_key"):
            st.markdown("### 🔑 API Key")

            key = st.text_input(
                "Hugging Face API Key",
                type="password",
                key="hf_key_input"
            )

            if key:
                st.session_state.hf_api_key = key
                st.success("API Key saved!")

            st.markdown("---")

        # Active Dataset Info
        active_dataset = st.session_state.get("active_dataset")

        if active_dataset:
            dataset_info = st.session_state.get("datasets", {}).get(
                active_dataset, {}
            )

            st.markdown(
                f"""
                <div style="
                    background:#1a2744;
                    border:1px solid #4f8ef7;
                    border-radius:8px;
                    padding:0.75rem;
                    margin-bottom:1rem;
                ">
                    <small>Active Dataset</small><br>
                    <strong>📂 {active_dataset}</strong><br>
                    <small>
                        {dataset_info.get('rows', 0):,} rows ·
                        {dataset_info.get('cols', 0)} columns
                    </small>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("No dataset loaded", icon="📭")

        st.markdown("### Navigation")

        # Navigation Buttons
        for label, page_id in PAGES.items():

            is_active = (
                st.session_state.active_page == page_id
            )

            if st.button(
                label,
                key=f"nav_{page_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.active_page = page_id
                st.rerun()

        # Dataset Switcher
        datasets = list(
            st.session_state.get("datasets", {}).keys()
        )

        if len(datasets) > 1:
            st.markdown("---")
            st.markdown("### Switch Dataset")

            current_dataset = st.session_state.get(
                "active_dataset"
            )

            default_index = (
                datasets.index(current_dataset)
                if current_dataset in datasets
                else 0
            )

            selected_dataset = st.selectbox(
                "Select Dataset",
                datasets,
                index=default_index,
            )

            if selected_dataset != current_dataset:
                st.session_state.active_dataset = (
                    selected_dataset
                )
                st.rerun()

        st.markdown("---")

        col1, col2 = st.columns(2)

        current_theme = st.session_state.get(
            "theme",
            "dark"
        )

        if col1.button(
            "☀️" if current_theme == "dark" else "🌙",
            use_container_width=True,
        ):
            st.session_state.theme = (
                "light"
                if current_theme == "dark"
                else "dark"
            )
            st.rerun()

        if col2.button(
            "🗑️",
            use_container_width=True,
            help="Clear Session",
        ):
            keys_to_remove = [
                "datasets",
                "query_history",
                "chat_messages",
                "dataset_versions",
            ]

            for key in keys_to_remove:
                if key in st.session_state:
                    del st.session_state[key]

            st.session_state.active_dataset = None
            st.rerun()

    # Route page
    _route_page(st.session_state.active_page)


def _route_page(page_id):
    """Load and render selected page."""

    if page_id == "dashboard":
        from app_pages.dashboard import render

    elif page_id == "upload":
        from app_pages.upload import render

    elif page_id == "cleaning":
        from app_pages.cleaning import render

    elif page_id == "explorer":
        from app_pages.explorer import render

    elif page_id == "ai_query":
        from app_pages.ai_query import render

    elif page_id == "visualizations":
        from app_pages.visualizations import render

    elif page_id == "insights":
        from app_pages.insights import render

    elif page_id == "forecasting":
        from app_pages.forecasting import render

    elif page_id == "anomalies":
        from app_pages.anomalies import render

    elif page_id == "reports":
        from app_pages.reports import render

    elif page_id == "settings":
        from app_pages.settings import render

    else:
        from app_pages.dashboard import render

    render()
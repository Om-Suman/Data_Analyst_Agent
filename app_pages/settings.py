"""Settings Page."""
import streamlit as st
from datetime import datetime


def render():
    st.title("⚙️ Settings")

    tab1, tab2, tab3, tab4 = st.tabs(["🔑 API & Models", "🎨 Appearance", "📊 Analysis", "🔄 Version History"])

    with tab1:
        st.markdown("### 🔑 Hugging Face API Configuration")
        current_key = st.session_state.get("hf_api_key", "")
        masked = f"{'*' * (len(current_key) - 4)}{current_key[-4:]}" if len(current_key) > 4 else "Not set"
        st.info(f"Current key: `{masked}`")

        new_key = st.text_input("New API Key", type="password", placeholder="hf_...")
        if st.button("Save API Key") and new_key:
            st.session_state.hf_api_key = new_key
            st.success("✅ API key updated!")

        st.markdown("---")
        st.markdown("### 🤖 Model Configuration")
        primary = st.text_input("Primary Model", value=st.session_state.get("primary_model", "deepseek-ai/DeepSeek-R1"))
        fallback = st.text_input("Fallback Model", value=st.session_state.get("fallback_model", ""))
        if st.button("Save Model Config"):
            st.session_state.primary_model = primary
            st.session_state.fallback_model = fallback
            st.success("✅ Model config saved!")

        st.markdown("---")
        st.markdown("### 📡 Inference Settings")
        max_tokens = st.slider("Max Tokens", 256, 4096, st.session_state.get("max_tokens", 2048), 256)
        temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.get("temperature", 0.3), 0.05)
        if st.button("Save Inference Settings"):
            st.session_state.max_tokens = max_tokens
            st.session_state.temperature = temperature
            st.success("✅ Saved!")

    with tab2:
        st.markdown("### 🎨 Appearance")
        theme = st.radio("Theme", ["dark", "light"], index=0 if st.session_state.get("theme", "dark") == "dark" else 1, horizontal=True)
        if theme != st.session_state.get("theme"):
            st.session_state.theme = theme
            st.rerun()

    with tab3:
        st.markdown("### 📊 Analysis Defaults")
        st.selectbox("Default Outlier Method", ["none", "zscore", "iqr"],
                     index=0, key="default_outlier")
        st.slider("Default Missing Threshold (% to drop column)", 0, 100, 50, key="default_missing_threshold")
        st.checkbox("Auto-profile on upload", value=False, key="auto_profile")
        st.checkbox("Auto-generate insights on upload", value=False, key="auto_insights")

    with tab4:
        st.markdown("### 🔄 Dataset Version History")
        active = st.session_state.get("active_dataset")
        if not active:
            st.info("No dataset loaded.")
            return

        versions = st.session_state.get("dataset_versions", {}).get(active, [])
        if not versions:
            st.info("No version history for this dataset.")
            return

        st.markdown(f"**{active}** — {len(versions)} version(s)")
        for v in reversed(versions):
            ts = v["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if isinstance(v["timestamp"], datetime) else str(v["timestamp"])
            with st.expander(f"v{v['version']} — {ts} — {v.get('description', '')}"):
                st.write(f"Rows: {v['rows']:,} | Cols: {v['cols']}")
                if st.button(f"Rollback to v{v['version']}", key=f"rollback_{v['version']}"):
                    snap_df = v.get("df_snapshot")
                    if snap_df is not None:
                        st.session_state.datasets[active]["df"] = snap_df.copy()
                        st.session_state.datasets[active]["rows"] = len(snap_df)
                        st.session_state.datasets[active]["cols"] = len(snap_df.columns)
                        st.success(f"✅ Rolled back to v{v['version']}")
                        st.rerun()

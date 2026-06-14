"""Configuration management."""
import streamlit as st
import os

def load_config():
    """Load app configuration into session state."""
    if "config_loaded" in st.session_state:
        return

    # API Key resolution priority: secrets → env → sidebar input
    api_key = None
    try:
        api_key = st.secrets["HF_API_KEY"]
    except Exception:
        api_key = os.environ.get("HF_API_KEY", "")

    st.session_state.hf_api_key = api_key

    # Model config
    st.session_state.primary_model = "Qwen/Qwen3-32B"
    st.session_state.fallback_model = "deepseek-ai/DeepSeek-R1"
    st.session_state.hf_inference_url = "https://api-inference.huggingface.co/v1/chat/completions"

    # App defaults
    st.session_state.theme = st.session_state.get("theme", "dark")
    st.session_state.max_tokens = st.session_state.get("max_tokens", 2048)
    st.session_state.temperature = st.session_state.get("temperature", 0.3)
    st.session_state.config_loaded = True

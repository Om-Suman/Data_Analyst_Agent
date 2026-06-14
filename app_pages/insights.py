"""Insights Center Page."""
import streamlit as st
import pandas as pd
from modules.insights import generate_statistical_insights, generate_llm_insights
from utils.session import get_active_df


def render():
    st.title("💡 Insights Center")
    df = get_active_df()
    if df is None:
        st.warning("No dataset loaded. Please upload data first.")
        return

    st.markdown("Generate AI-powered business insights from your data.")

    # Quick statistical insights (no LLM)
    st.markdown("### ⚡ Quick Statistical Insights")
    insights = generate_statistical_insights(df)
    for insight in insights:
        st.markdown(f"""<div class='insight-card'>{insight}</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # LLM Insights
    st.markdown("### 🤖 AI Business Intelligence")
    if not st.session_state.get("hf_api_key"):
        st.warning("Configure your Hugging Face API key to generate AI insights.")
        return

    if st.button("🚀 Generate AI Insights", type="primary"):
        with st.spinner("Analyzing data with AI..."):
            insights_data = generate_llm_insights(df, max_tokens=st.session_state.get("max_tokens", 1500))

        _display_insights(insights_data)

        # Store in session
        st.session_state["last_insights"] = insights_data

    elif st.session_state.get("last_insights"):
        st.info("Showing cached insights. Click button above to refresh.")
        _display_insights(st.session_state["last_insights"])


def _display_insights(data: dict):
    if not data:
        st.warning("No insights generated.")
        return

    # Executive Summary
    if data.get("executive_summary"):
        st.markdown("#### 📋 Executive Summary")
        st.info(data["executive_summary"])

    # Data Story
    if data.get("data_story"):
        st.markdown("#### 📖 Data Story")
        st.markdown(f"""<div class='insight-card'>{data['data_story']}</div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if data.get("key_findings"):
            st.markdown("#### 🔍 Key Findings")
            for item in data["key_findings"]:
                st.markdown(f"- {item}")

        if data.get("opportunities"):
            st.markdown("#### 🚀 Opportunities")
            for item in data["opportunities"]:
                st.markdown(f"- ✅ {item}")

    with col2:
        if data.get("trends"):
            st.markdown("#### 📈 Trends")
            for item in data["trends"]:
                st.markdown(f"- {item}")

        if data.get("risks"):
            st.markdown("#### ⚠️ Risks")
            for item in data["risks"]:
                st.markdown(f"- 🔴 {item}")

    if data.get("recommendations"):
        st.markdown("#### 💡 Recommendations")
        for i, rec in enumerate(data["recommendations"], 1):
            st.markdown(f"""
            <div class='insight-card'>
            <strong>{i}.</strong> {rec}
            </div>
            """, unsafe_allow_html=True)

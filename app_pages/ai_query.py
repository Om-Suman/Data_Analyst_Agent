"""AI Query Assistant Page."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from modules.query_engine import run_query
from utils.session import get_active_df, add_query_to_history
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUGGESTED_QUESTIONS = {
    "Analysis": [
        "What are the main patterns and trends in this data?",
        "Show statistical summary for all numeric columns",
        "Which category has the highest average value?",
        "What is the distribution of the top column?",
    ],
    "Visualization": [
        "Create a bar chart of top 10 values by the main metric",
        "Show a correlation heatmap of all numeric columns",
        "Plot the distribution of all numeric columns as histograms",
        "Create a scatter plot matrix for the numeric columns",
    ],
    "Business": [
        "Which region/category generated the highest revenue?",
        "Show month-over-month or period-over-period trends",
        "What are the top 10 records by the primary metric?",
        "Are there any outliers or anomalies worth investigating?",
    ],
    "Advanced": [
        "Perform a cohort analysis if time data is available",
        "Segment the data into meaningful groups and compare",
        "Calculate growth rates and percentage changes",
        "Generate an executive summary of key insights",
    ],
}


def render():
    st.title("🤖 AI Query Assistant")

    df = get_active_df()
    if df is None:
        st.warning("No dataset loaded. Please upload data first.")
        return

    name = st.session_state.active_dataset
    history = st.session_state.get("query_history", [])

    # API key check
    if not st.session_state.get("hf_api_key"):
        st.error("⚠️ No Hugging Face API key set. Please configure it in the sidebar.")
        return

    # Model info
    primary = st.session_state.get("primary_model", "Qwen/Qwen3-32B")
    fallback = st.session_state.get("fallback_model", "deepseek-ai/DeepSeek-R1")
    st.caption(f"Primary: `{primary}` | Fallback: `{fallback}`")

    # --- Suggested Questions ---
    with st.expander("💡 Suggested Questions", expanded=False):
        for category, questions in SUGGESTED_QUESTIONS.items():
            st.markdown(f"**{category}**")
            cols = st.columns(2)
            for i, q in enumerate(questions):
                if cols[i % 2].button(q, key=f"suggest_{category}_{i}", use_container_width=True):
                    st.session_state.selected_question = q

    # --- Question Input ---
    question = st.text_area(
        "Ask a question about your data:",
        value=st.session_state.get("selected_question", ""),
        height=80,
        placeholder="e.g. Which product had the highest sales last month? Show a bar chart.",
        key="query_input",
    )
    if "selected_question" in st.session_state:
        del st.session_state.selected_question

    col1, col2, col3 = st.columns([1, 1, 5])
    ask_btn = col1.button("🚀 Ask", type="primary", use_container_width=True)
    clear_btn = col2.button("🗑️ Clear", use_container_width=True)

    if clear_btn:
        st.session_state.query_history = []
        st.rerun()

    # --- Execute Query ---
    if ask_btn and question.strip():
        with st.spinner("🤔 Thinking..."):
            result = run_query(
                df=df,
                question=question,
                history=history,
                max_tokens=st.session_state.get("max_tokens", 2048),
            )

        if result.get("error"):
            st.error(result["error"])
        else:
            _display_result(result)
            # Save to history
            summary = result.get("insights", "")[:400]
            code_str = "\n\n".join(result.get("code_blocks", []))
            add_query_to_history(question, code_str, summary, name)

    # --- Chat History ---
    if history:
        st.markdown("---")
        st.markdown("### 🕒 Query History")
        for entry in reversed(history[-10:]):
            ts = entry["timestamp"].strftime("%H:%M:%S") if hasattr(entry.get("timestamp"), "strftime") else ""
            with st.expander(f"[{ts}] {entry['question'][:80]}", expanded=False):
                if entry.get("code"):
                    st.code(entry["code"], language="python")
                if entry.get("result_summary"):
                    st.markdown(entry["result_summary"])
                if st.button("🔄 Re-run", key=f"rerun_{entry['id']}"):
                    st.session_state.selected_question = entry["question"]
                    st.rerun()

        # Export history
        history_df = pd.DataFrame([
            {"Timestamp": e["timestamp"], "Question": e["question"],
             "Dataset": e.get("dataset", ""), "Code": e.get("code", "")}
            for e in history
        ])
        csv = history_df.to_csv(index=False).encode()
        st.download_button("📥 Export Query History", csv, "query_history.csv", "text/csv")


def _display_result(result: dict):
    model_used = result.get("model_used", "")
    st.markdown(f"<small style='color:#718096'>Model: {model_used}</small>", unsafe_allow_html=True)

    # Show LLM narrative (non-code parts)
    response = result.get("llm_response", "")
    # Strip code blocks for display
    import re
    narrative = re.sub(r"```python.*?```", "", response, flags=re.DOTALL).strip()
    if narrative:
        with st.expander("🧠 Analysis & Insights", expanded=True):
            st.markdown(narrative)

    # Execute and display code blocks
    exec_results = result.get("execution_results", [])
    code_blocks = result.get("code_blocks", [])

    for i, (code, exec_result) in enumerate(zip(code_blocks, exec_results)):
        with st.expander(f"🔧 Code Block {i+1}", expanded=(i == 0)):
            st.code(code, language="python")

        if exec_result.error:
            st.error(f"Execution error: {exec_result.error}")
        else:
            st.success(f"✅ Executed in {exec_result.execution_time:.2f}s")

            if exec_result.stdout:
                st.markdown("**Output:**")
                st.text(exec_result.stdout)

            # Plotly figures
            for fig in exec_result.figures:
                st.plotly_chart(fig, use_container_width=True)

            # Matplotlib figures
            for fig in exec_result.mpl_figures:
                st.pyplot(fig)
                plt.close(fig)

            # DataFrames created during execution
            for var_name, sub_df in exec_result.dataframes.items():
                st.markdown(f"**Created DataFrame: `{var_name}`**")
                st.dataframe(sub_df, use_container_width=True)

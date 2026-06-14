"""
AI Query Engine
Translates natural language into Pandas code and executes it safely.
"""
import pandas as pd
import streamlit as st
from modules.llm_client import query_llm, extract_python_code
from modules.executor import execute_code, ExecutionResult


SYSTEM_PROMPT = """You are an expert data analyst AI. You help users analyze datasets using Python/Pandas.

When asked to analyze data, you:
1. Write clean, optimized Pandas code using 'df' as the DataFrame variable
2. Generate Plotly visualizations (preferred) OR matplotlib when specified
3. Provide business insights in plain English
4. Explain what the code does

IMPORTANT:
- Never reveal chain-of-thought reasoning
- Never output <think> tags
- Never explain internal reasoning
- Output only Python code and Insights

RULES:
- Always use 'df' as the variable name
- For charts: use plotly express (px) or plotly graph_objects (go) — NOT plt.show() or fig.show()
- Assign plotly figures to variables named 'fig', 'fig1', 'fig2', etc.
- Use print() to output key findings/numbers
- Keep code clean, vectorized, and efficient
- Handle missing values gracefully
- Return code in ```python ... ``` blocks
- After the code block, provide a "## Insights" section with business-friendly interpretation

VISUALIZATION PREFERENCES:
- Bar: px.bar()
- Line: px.line()
- Scatter: px.scatter()
- Histogram: px.histogram()
- Box: px.box()
- Heatmap: px.imshow() for correlations, go.Heatmap() for custom
- Pie: px.pie()
- Use px.update_layout(template='plotly_dark') for styling
"""


def build_context(df: pd.DataFrame, question: str, history: list = None) -> str:
    """Build the user prompt with dataset context."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    # Sample data (first 5 rows, no more)
    sample_str = df.head(5).to_string(max_cols=15)

    # Stats only for numeric
    stats_str = ""
    if numeric_cols:
        stats_str = df[numeric_cols[:10]].describe().round(2).to_string()

    # Conversation context
    history_str = ""
    if history:
        history_str = "\n\nPrevious Q&A (last 3):\n"
        for entry in history[-3:]:
            history_str += f"Q: {entry['question']}\nA: {entry['result_summary'][:300]}\n\n"

    return f"""DATASET INFO:
- Shape: {df.shape[0]:,} rows × {df.shape[1]} columns
- Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:10])}
- Categorical columns ({len(cat_cols)}): {', '.join(cat_cols[:10])}
- Date columns ({len(date_cols)}): {', '.join(date_cols)}
- Missing values: {df.isnull().sum().sum():,}

SAMPLE DATA (first 5 rows):
{sample_str}

STATISTICAL SUMMARY:
{stats_str}
{history_str}

USER QUESTION: {question}

Generate Python code to answer this question and provide insights."""


def run_query(
    df: pd.DataFrame,
    question: str,
    history: list = None,
    max_tokens: int = 2048,
) -> dict:
    """
    Full pipeline: question → LLM → code → execute → result dict.
    Returns: {question, llm_response, code_blocks, execution_results, insights, model_used, error}
    """
    result = {
        "question": question,
        "llm_response": "",
        "code_blocks": [],
        "execution_results": [],
        "insights": "",
        "model_used": "",
        "error": None,
    }

    prompt = build_context(df, question, history)

    response, model_used = query_llm(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.3,
    )

    result["llm_response"] = response
    result["model_used"] = model_used

    if response.startswith("❌"):
        result["error"] = response
        return result

    # Extract code blocks
    code_blocks = extract_python_code(response)
    result["code_blocks"] = code_blocks

    # Extract insights (text after last code block)
    insights = response
    if "```" in response:
        parts = response.split("```")
        # Get text after last closing ```
        insights = parts[-1].strip()
    result["insights"] = insights

   
    exec_results = []
    for code in code_blocks:
        exec_result = execute_code(code, df)
        exec_results.append(exec_result)

    result["execution_results"] = exec_results

    return result

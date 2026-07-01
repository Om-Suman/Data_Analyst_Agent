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
3. Use print() to output the key facts needed to answer the question

IMPORTANT:
- Never reveal chain-of-thought reasoning
- Never output <think> tags
- Never explain internal reasoning
- Output only Python code in one or more ```python ... ``` blocks
- Do not write business insights or narrative in this first response

RULES:
- Always use 'df' as the variable name
- For charts: use plotly express (px) or plotly graph_objects (go) — NOT plt.show() or fig.show()
- Assign plotly figures to variables named 'fig', 'fig1', 'fig2', etc.
- Use print() to output key findings/numbers
- Keep code clean, vectorized, and efficient
- Handle missing values gracefully
- Never use .iloc[0], .iat[0], idxmax(), idxmin(), or positional indexing without first checking that the filtered DataFrame or Series is not empty
- When a filter can return no rows, print a clear message and create an empty result/table instead of raising an exception
- Return code in ```python ... ``` blocks

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


FINAL_INSIGHTS_SYSTEM_PROMPT = """You write final business insights after Pandas code has already executed.

Use only the execution results provided by the application. Do not invent rankings, counts, trends, or categories that are not present in stdout, generated tables, or errors.

Rules:
- If stdout says no rows or no matching records were found, clearly state that no ranking or exposure comparison can be computed.
- If execution failed, explain the failure plainly and avoid business conclusions.
- Keep the answer concise and business-friendly.
- Use markdown with a short "## Insights" heading.
- Do not include code.
"""


REPAIR_SYSTEM_PROMPT = """You repair failed Pandas analysis code.

Return only one corrected Python code block.

Rules:
- Use the existing DataFrame variable named df
- Do not load files or import modules
- Preserve the user's analysis intent
- Fix the exact runtime error
- Guard empty DataFrames/Series before using positional indexing, idxmax, or idxmin
- Use print() for user-facing messages
- Assign Plotly charts to fig, fig1, fig2, etc. when charts are needed
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

Generate Python code to answer this question. The app will execute the code and generate final insights afterward."""


def build_repair_prompt(
    df: pd.DataFrame,
    question: str,
    failed_code: str,
    error: str,
) -> str:
    """Build a prompt to repair generated code after a runtime failure."""
    context = build_context(df, question)
    return f"""{context}

The previous generated code failed at runtime.

FAILED CODE:
```python
{failed_code}
```

ERROR TRACEBACK:
{error[-3000:]}

Return corrected Python code only. Make it robust when filters return no rows."""


def build_execution_summary(exec_results: list[ExecutionResult]) -> str:
    """Summarize execution outputs for final insight generation."""
    sections = []

    for idx, exec_result in enumerate(exec_results, start=1):
        lines = [f"CODE BLOCK {idx}:"]
        lines.append(f"- Success: {exec_result.success}")

        if exec_result.error:
            lines.append("- Error:")
            lines.append(exec_result.error[-2500:])

        stdout = exec_result.stdout.strip()
        if stdout:
            lines.append("- Printed output:")
            lines.append(stdout[-3000:])
        else:
            lines.append("- Printed output: None")

        if exec_result.dataframes:
            lines.append("- Created DataFrames:")
            for name, sub_df in exec_result.dataframes.items():
                lines.append(f"  - {name}: {sub_df.shape[0]} rows x {sub_df.shape[1]} columns")
                preview = sub_df.head(10).to_string(max_cols=8, max_colwidth=40)
                lines.append(preview)
        else:
            lines.append("- Created DataFrames: None")

        lines.append(f"- Plotly figures: {len(exec_result.figures)}")
        lines.append(f"- Matplotlib figures: {len(exec_result.mpl_figures)}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def build_final_insights_prompt(
    question: str,
    code_blocks: list[str],
    exec_results: list[ExecutionResult],
    repaired_any: bool,
) -> str:
    """Build a grounded final-insights prompt from actual execution results."""
    code_text = "\n\n".join(
        f"CODE BLOCK {idx}:\n```python\n{code}\n```"
        for idx, code in enumerate(code_blocks, start=1)
    )
    repair_note = "Yes" if repaired_any else "No"
    execution_summary = build_execution_summary(exec_results)

    return f"""USER QUESTION:
{question}

EXECUTED CODE:
{code_text}

WAS ANY CODE AUTO-REPAIRED:
{repair_note}

ACTUAL EXECUTION RESULTS:
{execution_summary}

Write final insights that are strictly supported by the actual execution results."""


def build_fallback_insights(exec_results: list[ExecutionResult]) -> str:
    """Create deterministic insights when the final LLM call is unavailable."""
    outputs = []
    errors = []
    for exec_result in exec_results:
        if exec_result.stdout.strip():
            outputs.append(exec_result.stdout.strip())
        if exec_result.error:
            errors.append(exec_result.error.strip().splitlines()[-1])

    if errors:
        return "## Insights\nThe analysis code did not complete successfully, so no business conclusion should be drawn from this run."
    if outputs:
        return "## Insights\n" + "\n\n".join(outputs)
    return "## Insights\nThe code executed successfully, but it did not print enough information to generate a grounded narrative."


def generate_final_insights(
    question: str,
    code_blocks: list[str],
    exec_results: list[ExecutionResult],
    repaired_any: bool,
    max_tokens: int,
) -> tuple[str, str]:
    """Generate final insights from actual execution outputs."""
    prompt = build_final_insights_prompt(question, code_blocks, exec_results, repaired_any)
    response, model_used = query_llm(
        system_prompt=FINAL_INSIGHTS_SYSTEM_PROMPT,
        user_prompt=prompt,
        max_tokens=min(max_tokens, 1200),
        temperature=0.2,
        retries=1,
        timeout=60,
    )

    if response.startswith("âŒ") or response.startswith("❌") or not response.strip():
        return build_fallback_insights(exec_results), "fallback"

    return response.strip(), model_used


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
        "code_generation_response": "",
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

    result["code_generation_response"] = response
    result["llm_response"] = ""
    result["model_used"] = model_used

    if response.startswith("❌"):
        result["error"] = response
        return result

    # Extract code blocks
    code_blocks = extract_python_code(response)
    result["code_blocks"] = code_blocks

    result["insights"] = ""

   
    exec_results = []
    repaired_any = False
    final_code_blocks = []
    for code in code_blocks:
        exec_result = execute_code(code, df)
        final_code = code

        if exec_result.error:
            repair_prompt = build_repair_prompt(
                df=df,
                question=question,
                failed_code=code,
                error=exec_result.error,
            )
            repair_response, repair_model = query_llm(
                system_prompt=REPAIR_SYSTEM_PROMPT,
                user_prompt=repair_prompt,
                max_tokens=max_tokens,
                temperature=0.1,
            )
            repaired_blocks = extract_python_code(repair_response)
            if repaired_blocks:
                repaired_result = execute_code(repaired_blocks[0], df)
                if repaired_result.success:
                    exec_result = repaired_result
                    final_code = repaired_blocks[0]
                    repaired_any = True
                    result["model_used"] = f"{model_used} + repair:{repair_model}"

        final_code_blocks.append(final_code)
        exec_results.append(exec_result)

    if repaired_any:
        result["code_blocks"] = final_code_blocks
        repair_note = "\n\n## Repair Note\nA generated code block raised an execution error, so it was automatically corrected and re-run."

    result["execution_results"] = exec_results

    final_insights, insights_model = generate_final_insights(
        question=question,
        code_blocks=final_code_blocks,
        exec_results=exec_results,
        repaired_any=repaired_any,
        max_tokens=max_tokens,
    )
    if repaired_any:
        final_insights = (final_insights + repair_note).strip()

    result["insights"] = final_insights
    result["llm_response"] = final_insights
    if result["model_used"]:
        result["model_used"] = f"{result['model_used']} + insights:{insights_model}"
    else:
        result["model_used"] = f"insights:{insights_model}"

    return result



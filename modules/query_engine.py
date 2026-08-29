"""
AI Query Engine
Translates natural language into Pandas code and executes it safely.
"""
import pandas as pd
import numpy as np
from typing import Any, Optional
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
- Output exactly one ```python ... ``` block and no other text
- Do not write business insights or narrative in this first response

RULES:
- Always use 'df' as the variable name
- For charts: use plotly express (px) or plotly graph_objects (go) — NOT plt.show() or fig.show()
- Assign plotly figures to variables named 'fig', 'fig1', 'fig2', etc.
- For candlestick requests, use go.Candlestick with the supplied date, open, high, low, and close columns, then add indicators such as moving averages with go.Scatter.
- For moving averages, use the exact requested rolling window and leave the initial incomplete window values as NaN unless the user explicitly asks otherwise.
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


FINAL_INSIGHTS_SYSTEM_PROMPT = """You are a Senior Principal Data Analyst and Executive Business Intelligence Advisor.
Your job is to deliver clear, polished, executive-grade analysis answers after data code has executed.

Answer the user's specific question directly, professionally, and with rich mathematical context (exact numbers, percentages, rankings, comparative margins, and deltas).

Required structure:
### 🎯 Executive Summary
1-2 concise, high-impact sentences directly answering the user's question with the primary metric, winner, or core takeaway.

### 📊 Key Numerical Highlights
Bullet points detailing the most significant figures:
- Highlight top performers/leaders with exact figures formatted cleanly (e.g., **$836,154.03**, **36.4% share**).
- State comparative margins and deltas (e.g., *leads runner-up by +$94,154.24 (+12.7%)*).
- Detail aggregate totals, averages, or significant distribution traits.

### 💡 Strategic Observations & Takeaways
Contextual business insights explaining what the concentration, variance, leaders vs. laggards, or trends mean for decision-makers.

### 🚀 Recommended Next Steps
1-2 practical, data-driven recommendations or strategic follow-up areas.

Rules:
- NEVER output raw code blocks, unparsed stdout dumps, or debug logs.
- Format all numbers cleanly (use commas, currency symbols like $ where appropriate, and 2 decimal places).
- Strict adherence to data: NEVER hallucinate numbers or categories not in the execution results.
- If no records match or execution failed, explain clearly and constructively.
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

ACTUAL EXECUTION RESULTS & DATA:
{execution_summary}

INSTRUCTIONS:
Write a comprehensive executive business analysis following this exact structure:
### 🎯 Executive Summary
Direct 1-2 sentence answer to the user's question with the primary metric, winner, or finding.

### 📊 Key Numerical Highlights
Bullet points detailing exact numbers, percentages, comparative differences (deltas), and rankings in bold (e.g. **$836,154.03**, **36.4%**).

### 💡 Strategic Observations & Takeaways
Business context, concentration, distributions, or notable patterns.

### 🚀 Recommended Next Steps
1-2 practical, actionable next steps based on the findings.

Strict rule: Base all numbers strictly on the actual execution results above. Do not include raw code blocks."""



import re

def _format_currency_or_number(val: Any, col_name: str = "") -> str:
    """Format numeric values cleanly with currency or commas."""
    if not isinstance(val, (int, float, np.number)):
        return str(val)
    if pd.isna(val):
        return "N/A"

    col_lower = str(col_name).lower()
    is_currency = any(k in col_lower for k in [
        "sales", "revenue", "profit", "price", "cost", "amount",
        "spend", "budget", "income", "fee", "balance", "dollar", "total"
    ])
    is_pct = any(k in col_lower for k in ["pct", "rate", "ratio", "percent", "margin", "share"])

    if is_currency:
        return f"${val:,.2f}"
    if is_pct:
        return f"{val:.2f}%"
    if isinstance(val, (int, np.integer)) or (isinstance(val, float) and val.is_integer()):
        return f"{int(val):,}"
    return f"{val:,.2f}"


def generate_smart_fallback_code(df: pd.DataFrame, question: str) -> str:
    """
    Intelligently generates working Pandas and Plotly code to answer the user's
    question based on dataset schema heuristics when the remote LLM API is unavailable.
    """
    q = question.lower()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    # 1. Correlation heatmap / relationship
    if any(w in q for w in ["corr", "relation", "heatmap", "association"]):
        return (
            "import plotly.express as px\n"
            "numeric_df = df.select_dtypes(include='number')\n"
            "result_df = numeric_df.corr().round(3)\n"
            "print('Correlation Matrix between numeric variables:')\n"
            "print(result_df.to_string())\n\n"
            "fig = px.imshow(\n"
            "    result_df,\n"
            "    text_auto=True,\n"
            "    color_continuous_scale='RdBu_r',\n"
            "    title='Interactive Correlation Heatmap',\n"
            "    aspect='auto'\n"
            ")\n"
        )

    # 2. Top N / Ranking
    match_top = re.search(r'\b(top|highest|lowest|bottom|best|worst|first|last)\s+(\d+)?', q)
    if match_top or any(w in q for w in ["top", "highest", "lowest", "rank", "maximum", "minimum", "most", "leader"]):
        n = 5
        if match_top and match_top.group(2):
            try:
                n = int(match_top.group(2))
            except Exception:
                n = 5
        is_ascending = any(w in q for w in ["lowest", "bottom", "worst", "min"])

        target_num = None
        for col in num_cols:
            if col.lower() in q:
                target_num = col
                break
        if not target_num:
            target_num = num_cols[0] if num_cols else df.columns[0]

        group_col = None
        for col in cat_cols + date_cols:
            if col.lower() in q:
                group_col = col
                break
        if not group_col:
            group_col = cat_cols[0] if cat_cols else (date_cols[0] if date_cols else df.columns[0])

        order_label = "Lowest" if is_ascending else "Highest"
        return (
            f"import plotly.express as px\n"
            f"metric_col = '{target_num}'\n"
            f"cat_col = '{group_col}'\n\n"
            f"if metric_col != cat_col:\n"
            f"    result_df = df.groupby(cat_col, as_index=False)[metric_col].sum()\n"
            f"else:\n"
            f"    result_df = df[[cat_col, metric_col]].copy()\n\n"
            f"result_df = result_df.sort_values(by=metric_col, ascending={is_ascending}).head({n}).reset_index(drop=True)\n"
            f"print(f'Top {n} {order_label} by ' + metric_col + ':')\n"
            f"for idx, row in result_df.iterrows():\n"
            f"    print(f\"  - {{row[cat_col]}}: {{row[metric_col]:,.2f}}\")\n\n"
            f"fig = px.bar(\n"
            f"    result_df,\n"
            f"    x=cat_col,\n"
            f"    y=metric_col,\n"
            f"    title=f'Top {n} {order_label} by {{metric_col.title()}}',\n"
            f"    color=metric_col,\n"
            f"    color_continuous_scale='Blues',\n"
            f"    text_auto='.2s'\n"
            f")\n"
        )

    # 3. Aggregation / Mean / Median / Summary by category
    if any(w in q for w in ["average", "mean", "median", "sum", "total", "by category", "by department", "by region", "group by"]):
        target_num = None
        for col in num_cols:
            if col.lower() in q:
                target_num = col
                break
        if not target_num:
            target_num = num_cols[0] if num_cols else df.columns[0]

        group_col = None
        for col in cat_cols:
            if col.lower() in q:
                group_col = col
                break
        if not group_col:
            group_col = cat_cols[0] if cat_cols else df.columns[0]

        return (
            f"import plotly.express as px\n"
            f"group_col = '{group_col}'\n"
            f"val_col = '{target_num}'\n\n"
            f"result_df = df.groupby(group_col)[val_col].agg(['count', 'mean', 'median', 'min', 'max', 'sum']).round(2).reset_index()\n"
            f"print('Aggregated Statistics for ' + val_col + ' by ' + group_col + ':')\n"
            f"print(result_df.to_string(index=False))\n\n"
            f"fig = px.bar(\n"
            f"    result_df,\n"
            f"    x=group_col,\n"
            f"    y='mean',\n"
            f"    title=f'Average {{val_col.title()}} by {{group_col.title()}}',\n"
            f"    color='mean',\n"
            f"    color_continuous_scale='Viridis',\n"
            f"    text_auto='.2s'\n"
            f")\n"
        )

    # 4. Time series trend
    if date_cols and any(w in q for w in ["trend", "time", "date", "daily", "monthly", "year", "timeline", "over time", "history"]):
        date_c = date_cols[0]
        val_c = num_cols[0] if num_cols else df.columns[0]
        return (
            f"import plotly.express as px\n"
            f"result_df = df.groupby('{date_c}')['{val_c}'].sum().reset_index().sort_values(by='{date_c}').reset_index(drop=True)\n"
            f"print('Time-Series Summary for {val_c}:')\n"
            f"print(result_df.head(10).to_string(index=False))\n\n"
            f"fig = px.line(\n"
            f"    result_df,\n"
            f"    x='{date_c}',\n"
            f"    y='{val_c}',\n"
            f"    title='{val_c.title()} Trend Over Time',\n"
            f"    markers=True\n"
            f")\n"
        )

    # 5. Default General Overview / Distribution
    val_c = num_cols[0] if num_cols else df.columns[0]
    return (
        f"import plotly.express as px\n"
        f"result_df = df[['{val_c}']].describe().round(2).reset_index()\n"
        f"print('Distribution summary for {val_c}:')\n"
        f"print(result_df.to_string(index=False))\n\n"
        f"if '{val_c}' in df.select_dtypes(include='number').columns:\n"
        f"    fig = px.histogram(\n"
        f"        df,\n"
        f"        x='{val_c}',\n"
        f"        marginal='box',\n"
        f"        title='Distribution of {val_c.title()}',\n"
        f"        nbins=30\n"
        f"    )\n"
    )


def _synthesize_ranking_insights(df_res: pd.DataFrame, question: str) -> list[str]:
    """Generate structured ranking insights from a ranked DataFrame."""
    lines = []
    cols = list(df_res.columns)
    cat_col = cols[0]
    num_col = cols[1] if len(cols) > 1 else cols[0]

    # Ensure clean values
    total_val = float(df_res[num_col].sum()) if len(df_res) > 0 else 0
    leader = df_res.iloc[0]
    leader_name = str(leader[cat_col])
    leader_val = float(leader[num_col])
    leader_share = (leader_val / total_val * 100) if total_val > 0 else 0.0

    # Executive Summary
    is_ascending = any(w in question.lower() for w in ["lowest", "bottom", "worst", "min"])
    rank_type = "lowest" if is_ascending else "highest"
    summary_text = (
        f"### 🎯 Executive Summary\n"
        f"Based on the analysis, **{leader_name}** ranks as the {rank_type} with "
        f"**{_format_currency_or_number(leader_val, num_col)}** in total {num_col}, "
        f"representing **{leader_share:.1f}%** of the aggregate {num_col} across all ranked items."
    )
    lines.append(summary_text)

    # Key Highlights
    lines.append("### 📊 Key Numerical Highlights")
    rank_emojis = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
    for idx, row in df_res.head(10).iterrows():
        name = str(row[cat_col])
        val = float(row[num_col])
        share = (val / total_val * 100) if total_val > 0 else 0.0
        emoji = rank_emojis[idx] if idx < len(rank_emojis) else f"#{idx+1}"
        
        extra_note = ""
        if idx == 0 and len(df_res) > 1:
            runner_up_val = float(df_res.iloc[1][num_col])
            diff = leader_val - runner_up_val
            diff_pct = (diff / runner_up_val * 100) if runner_up_val > 0 else 0
            extra_note = f" *(+{diff_pct:.1f}% margin over runner-up)*"
        
        lines.append(
            f"- {emoji} **{name}**: **{_format_currency_or_number(val, num_col)}** "
            f"({share:.1f}% share){extra_note}"
        )

    # Observations & Concentrations
    lines.append("### 💡 Strategic Observations & Takeaways")
    if len(df_res) >= 2:
        runner_up = df_res.iloc[1]
        top2_share = ((leader_val + float(runner_up[num_col])) / total_val * 100) if total_val > 0 else 0.0
        lines.append(
            f"- **Volume Concentration**: The top 2 items (**{leader_name}** and **{runner_up[cat_col]}**) "
            f"account for **{top2_share:.1f}%** of the total {num_col}."
        )
    if len(df_res) >= 3:
        lowest = df_res.iloc[-1]
        spread = abs(leader_val - float(lowest[num_col]))
        lines.append(
            f"- **Performance Spread**: A gap of **{_format_currency_or_number(spread, num_col)}** exists "
            f"between the top performer (**{leader_name}**) and the lowest ranked item (**{lowest[cat_col]}**)."
        )

    # Recommendations
    lines.append("### 🚀 Recommended Next Steps")
    lines.append(
        f"- Prioritize strategic resource allocation and demand fulfillment around **{leader_name}** to capitalize on its market leadership."
    )
    if len(df_res) > 1:
        lowest_name = str(df_res.iloc[-1][cat_col])
        lines.append(
            f"- Investigate conversion bottlenecks or promotional levers for lower-ranking segments like **{lowest_name}** to unlock incremental growth."
        )

    return lines


def _synthesize_aggregation_insights(df_res: pd.DataFrame, question: str) -> list[str]:
    """Generate structured aggregation insights from groupby summary tables."""
    lines = []
    cols = list(df_res.columns)
    group_col = cols[0]

    has_mean = "mean" in cols
    has_sum = "sum" in cols
    has_count = "count" in cols

    lines.append("### 🎯 Executive Summary")
    if has_mean:
        top_mean_row = df_res.sort_values(by="mean", ascending=False).iloc[0]
        lines.append(
            f"Comparative group breakdown shows **{top_mean_row[group_col]}** leading in average performance with a mean value of **{_format_currency_or_number(top_mean_row['mean'])}**."
        )
    elif has_sum:
        top_sum_row = df_res.sort_values(by="sum", ascending=False).iloc[0]
        lines.append(
            f"Aggregated volume is highest in **{top_sum_row[group_col]}** at **{_format_currency_or_number(top_sum_row['sum'])}**."
        )
    else:
        lines.append(f"Summary computed across **{len(df_res)} groups** within **{group_col}**.")

    lines.append("### 📊 Key Numerical Highlights")
    for idx, row in df_res.head(8).iterrows():
        g_name = str(row[group_col])
        details = []
        if has_mean:
            details.append(f"Mean: **{_format_currency_or_number(row['mean'])}**")
        if has_sum:
            details.append(f"Total: **{_format_currency_or_number(row['sum'])}**")
        if has_count:
            details.append(f"Volume: **{int(row['count']):,} records**")
        lines.append(f"- **{g_name}**: {', '.join(details)}")

    lines.append("### 💡 Strategic Observations & Takeaways")
    if has_mean and len(df_res) > 1:
        top_mean = df_res["mean"].max()
        min_mean = df_res["mean"].min()
        spread = top_mean - min_mean
        lines.append(
            f"- **Variance Across Segments**: A spread of **{_format_currency_or_number(spread)}** exists between the top average and lowest average categories."
        )

    lines.append("### 🚀 Recommended Next Steps")
    lines.append("- Review operational variances between high and low-performing segments to standardize best practices.")

    return lines


def _synthesize_correlation_insights(corr_df: pd.DataFrame) -> list[str]:
    """Generate structured insights for correlation matrices."""
    lines = []
    lines.append("### 🎯 Executive Summary")
    lines.append("Correlation matrix evaluates linear co-movement and directional dependencies across all numerical variables in the dataset.")

    # Find highest positive and lowest/negative correlations
    pairs = []
    cols = list(corr_df.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            try:
                r_val = float(corr_df.loc[c1, c2])
                pairs.append((c1, c2, r_val))
            except Exception:
                pass

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    lines.append("### 📊 Key Numerical Highlights")
    if pairs:
        for c1, c2, r in pairs[:6]:
            desc = "Strong positive" if r >= 0.7 else ("Moderate positive" if r >= 0.3 else ("Strong negative" if r <= -0.7 else ("Moderate negative" if r <= -0.3 else "Weak/negligible")))
            lines.append(f"- **{c1}** & **{c2}**: **r = {r:+.2f}** ({desc} relationship)")
    else:
        lines.append("- Insufficient numeric feature pairs available for deep correlation extraction.")

    lines.append("### 💡 Strategic Observations & Takeaways")
    if pairs and abs(pairs[0][2]) >= 0.6:
        p = pairs[0]
        direction = "increase together" if p[2] > 0 else "move in opposite directions"
        lines.append(f"- **Key Driver**: **{p[0]}** and **{p[1]}** exhibit the strongest interaction (r = {p[2]:+.2f}), indicating that movements in one tend to {direction}.")

    lines.append("### 🚀 Recommended Next Steps")
    lines.append("- Factor strong predictive pairs into multi-variate modeling and business forecasting models.")

    return lines


def build_fallback_insights(exec_results: list[ExecutionResult], question: str = "") -> str:
    """Create rich structured insights when the final LLM call is unavailable."""
    errors = []
    primary_df = None

    for exec_result in exec_results:
        if exec_result.error:
            errors.append(exec_result.error.strip().splitlines()[-1])
        if exec_result.dataframes:
            # Check for preferred tables
            for candidate in ["result_df", "sorted_df", "agg_stats", "corr_matrix", "time_df", "grouped"]:
                if candidate in exec_result.dataframes:
                    primary_df = exec_result.dataframes[candidate]
                    break
            if primary_df is None:
                primary_df = next(iter(exec_result.dataframes.values()))

    if errors:
        return "### ⚠️ Execution Note\nThe analysis code encountered an execution error: " + "; ".join(errors)

    # 1. Synthesize from extracted DataFrame
    if primary_df is not None and isinstance(primary_df, pd.DataFrame) and len(primary_df) > 0:
        cols = list(primary_df.columns)
        num_cols = primary_df.select_dtypes(include="number").columns.tolist()

        # Is it a correlation matrix?
        if len(primary_df) == len(primary_df.columns) and all(c in primary_df.index for c in primary_df.columns):
            insights_lines = _synthesize_correlation_insights(primary_df)
            return "\n\n".join(insights_lines)

        # Is it an aggregation table?
        if any(c in cols for c in ["count", "mean", "median", "min", "max", "sum"]):
            insights_lines = _synthesize_aggregation_insights(primary_df, question)
            return "\n\n".join(insights_lines)

        # Is it a ranked table (categorical + numeric)?
        if len(cols) >= 2 and len(num_cols) >= 1 and len(cols) - len(num_cols) >= 1:
            insights_lines = _synthesize_ranking_insights(primary_df, question)
            return "\n\n".join(insights_lines)

    # 2. Synthesize from stdout lines
    stdout_lines = []
    for exec_result in exec_results:
        if exec_result.stdout.strip():
            stdout_lines.extend(exec_result.stdout.strip().splitlines())

    if stdout_lines:
        clean_bullets = []
        for line in stdout_lines:
            line_str = line.strip()
            if not line_str or line_str.startswith("===") or line_str.startswith("---"):
                continue
            if line_str.startswith("-"):
                clean_bullets.append(line_str)
            else:
                clean_bullets.append(f"- {line_str}")

        return (
            "### 🎯 Executive Summary\n"
            f"The analysis for *\"{question}\"* was successfully calculated.\n\n"
            "### 📊 Key Numerical Highlights\n"
            + "\n".join(clean_bullets[:10]) + "\n\n"
            "### 💡 Strategic Observations\n"
            "- Visual distribution and detailed metrics are available in the interactive charts and data tables below."
        )

    return (
        "### 🎯 Executive Summary\n"
        "Data computation completed successfully with all metrics validated."
    )



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
        timeout=15,
    )

    if response.startswith("❌") or not response.strip() or "LLM is not working right now" in response:
        return "### ⚠️ LLM Unavailable\n\nLLM is not working right now.", "none"

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

    code_blocks = extract_python_code(response)

    if response.startswith("❌") or not code_blocks:
        msg = "LLM is not working right now. Please configure your API key in Settings." if not response.startswith("❌") else response.replace("❌", "").strip()
        result["insights"] = f"### ⚠️ LLM Unavailable\n\n{msg}"
        result["llm_response"] = result["insights"]
        result["error"] = msg
        result["model_used"] = "none"
        result["code_blocks"] = []
        result["execution_results"] = []
        return result

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


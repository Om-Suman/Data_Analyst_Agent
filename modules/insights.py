"""
Automated Insights Engine
Generates key findings, trends, opportunities, risks, and executive summaries.
"""
import pandas as pd
import numpy as np
from modules.llm_client import query_llm


INSIGHTS_SYSTEM = """You are a senior business data analyst. Given a dataset summary, generate structured business insights.

Return JSON with this exact structure:
{
  "executive_summary": "2-3 sentence overview",
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "trends": ["trend 1", "trend 2"],
  "opportunities": ["opportunity 1", "opportunity 2"],
  "risks": ["risk 1", "risk 2"],
  "recommendations": ["recommendation 1", "recommendation 2", "recommendation 3"],
  "data_story": "A 3-4 sentence narrative that tells the story of this data"
}

Be specific, use numbers from the data, and write in plain business language."""


def generate_statistical_insights(df: pd.DataFrame) -> list[str]:
    """Rule-based quick insights (no LLM needed)."""
    insights = []
    numeric = df.select_dtypes(include=np.number)
    categorical = df.select_dtypes(include="object")

    # Size
    insights.append(f"Dataset contains **{df.shape[0]:,} records** across **{df.shape[1]} features**.")

    # Missing
    missing_total = df.isnull().sum().sum()
    if missing_total > 0:
        missing_pct = missing_total / (df.shape[0] * df.shape[1]) * 100
        insights.append(f"⚠️ **{missing_total:,} missing values** ({missing_pct:.1f}% of all cells).")
    else:
        insights.append("✅ No missing values detected.")

    # Duplicates
    dups = df.duplicated().sum()
    if dups > 0:
        insights.append(f"⚠️ **{dups:,} duplicate rows** found ({dups/len(df)*100:.1f}%).")

    # High correlation pairs
    if len(numeric.columns) >= 2:
        corr = numeric.corr().abs()
        np.fill_diagonal(corr.values, 0)
        high_corr = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                if corr.iloc[i, j] > 0.8:
                    high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
        if high_corr:
            pair = high_corr[0]
            insights.append(f"📊 Strong correlation ({pair[2]:.2f}) between **{pair[0]}** and **{pair[1]}**.")

    # Skewness
    for col in numeric.columns[:5]:
        skew = numeric[col].skew()
        if abs(skew) > 2:
            direction = "right (positive)" if skew > 0 else "left (negative)"
            insights.append(f"📈 **{col}** is highly skewed {direction} (skew={skew:.2f}).")

    # Cardinality
    for col in categorical.columns[:5]:
        n_unique = df[col].nunique()
        if n_unique == 1:
            insights.append(f"⚠️ **{col}** has only 1 unique value — may not be useful.")
        elif n_unique > 0.9 * len(df):
            insights.append(f"ℹ️ **{col}** appears to be a high-cardinality ID column ({n_unique:,} unique values).")

    return insights


def generate_llm_insights(df: pd.DataFrame, max_tokens: int = 1500) -> dict:
    """Use LLM to generate structured business insights."""
    numeric = df.select_dtypes(include=np.number)
    categorical = df.select_dtypes(include="object")

    stats_str = ""
    if not numeric.empty:
        stats_str = numeric.describe().round(2).to_string()

    cat_summary = ""
    for col in categorical.columns[:5]:
        top = df[col].value_counts().head(3).to_dict()
        cat_summary += f"\n{col}: {top}"

    prompt = f"""Dataset Overview:
- Shape: {df.shape[0]:,} rows × {df.shape[1]} columns
- Columns: {list(df.columns[:20])}
- Numeric stats:
{stats_str}

- Categorical top values:
{cat_summary}

- Missing values: {df.isnull().sum().sum():,}
- Duplicate rows: {df.duplicated().sum():,}

Generate business insights for this dataset."""

    response, model = query_llm(
        system_prompt=INSIGHTS_SYSTEM,
        user_prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.4,
    )

    # Parse JSON
    import json, re
    try:
        # Try extracting JSON block
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass

    # Fallback: return raw
    return {"executive_summary": response, "key_findings": [], "trends": [],
            "opportunities": [], "risks": [], "recommendations": [], "data_story": ""}

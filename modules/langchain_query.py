"""LangChain-backed orchestration for dataframe questions."""

from __future__ import annotations

import re

import pandas as pd

from modules.executor import execute_code
from modules.llm_client import extract_json, extract_python_code, query_llm
from modules.anomaly_detection import detect_iqr, detect_isolation_forest, detect_zscore
from modules.document_rag import answer_document_question
from modules.forecasting import (
    exponential_smoothing_forecast,
    linear_trend_forecast,
    moving_average_forecast,
)
from modules.insights import generate_statistical_insights
from modules.query_engine import (
    REPAIR_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_context,
    build_repair_prompt,
    generate_final_insights,
)

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda

    LANGCHAIN_AVAILABLE = True
except Exception:
    ChatPromptTemplate = None
    RunnableLambda = None
    LANGCHAIN_AVAILABLE = False


ROUTER_SYSTEM_PROMPT = """You route data-analysis questions to one safe capability.

Return only JSON in this exact shape:
{
  "intent": "dataframe_analysis|statistical_summary|forecast|anomaly_detection|document_qa",
  "reason": "short explanation",
  "column": "an explicitly mentioned column name, or empty string",
  "method": "linear_trend|moving_average|exponential_smoothing|isolation_forest|zscore|iqr, or empty string"
}

Choose dataframe_analysis for aggregation, filtering, comparisons, and charts.
Choose statistical_summary for broad descriptive summaries or high-level patterns.
Choose forecast only when the user asks for a future projection.
Choose anomaly_detection only when the user asks for anomalies, outliers, or unusual records.
Choose document_qa only when the active input is a document.
Do not invent column names."""

VALID_INTENTS = {
    "dataframe_analysis",
    "statistical_summary",
    "forecast",
    "anomaly_detection",
    "document_qa",
}


def langchain_available() -> bool:
    return LANGCHAIN_AVAILABLE


def _heuristic_route(question: str, is_document: bool = False) -> dict:
    """Safe fallback when an LLM route cannot be parsed or is unavailable."""
    text = question.lower()
    if is_document:
        intent = "document_qa"
    elif any(word in text for word in ("forecast", "predict", "projection", "next month", "next quarter", "future")):
        intent = "forecast"
    elif any(word in text for word in ("anomal", "outlier", "unusual", "fraud")):
        intent = "anomaly_detection"
    elif any(phrase in text for phrase in ("statistical summary", "describe the data", "main patterns", "overview of the data")):
        intent = "statistical_summary"
    else:
        intent = "dataframe_analysis"
    return {"intent": intent, "reason": "Keyword fallback routing.", "column": "", "method": ""}


def _build_router_chain():
    if not LANGCHAIN_AVAILABLE:
        return None

    prompt = ChatPromptTemplate.from_messages(
        [("system", "{system_prompt}"), ("user", "{user_prompt}")]
    )

    def _call_router(prompt_value):
        messages = prompt_value.to_messages()
        response, model = query_llm(
            system_prompt=messages[0].content,
            user_prompt=messages[1].content,
            max_tokens=300,
            temperature=0.0,
        )
        return {"response": response, "model": model}

    return prompt | RunnableLambda(_call_router)


def route_question(
    question: str,
    df: pd.DataFrame | None = None,
    is_document: bool = False,
) -> dict:
    """Classify a question into a constrained application capability."""
    fallback = _heuristic_route(question, is_document=is_document)
    schema = "document input" if is_document else (
        ", ".join(f"{name} ({dtype})" for name, dtype in df.dtypes.items()) if df is not None else "no dataset"
    )
    available_columns = "none" if is_document else schema
    user_prompt = f"""Question: {question}
Input type: {schema}
Available columns: {available_columns}"""

    if not LANGCHAIN_AVAILABLE:
        return {**fallback, "routing_source": "heuristic", "routing_model": "none"}

    try:
        payload = _build_router_chain().invoke(
            {"system_prompt": ROUTER_SYSTEM_PROMPT, "user_prompt": user_prompt}
        )
        parsed = extract_json(payload["response"])
        if not isinstance(parsed, dict) or parsed.get("intent") not in VALID_INTENTS:
            raise ValueError("Router returned an invalid intent.")
        if is_document:
            parsed["intent"] = "document_qa"
        elif parsed["intent"] == "document_qa":
            parsed["intent"] = "dataframe_analysis"
        parsed.setdefault("reason", "LLM-selected capability.")
        parsed.setdefault("column", "")
        parsed.setdefault("method", "")
        return {**parsed, "routing_source": "langchain", "routing_model": payload["model"]}
    except Exception:
        return {**fallback, "routing_source": "heuristic", "routing_model": "none"}


def _select_numeric_column(df: pd.DataFrame, requested_column: str) -> str:
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    if not numeric_columns:
        raise ValueError("This operation needs at least one numeric column.")
    requested = str(requested_column or "").strip().lower()
    for column in numeric_columns:
        if column.lower() == requested:
            return column
    for column in numeric_columns:
        if requested and requested in column.lower():
            return column
    return numeric_columns[0]


def _forecast_horizon(question: str) -> int:
    match = re.search(r"(?:next|for)\s+(\d+)\s+(?:periods?|days?|weeks?|months?)", question.lower())
    return max(1, min(int(match.group(1)), 365)) if match else 30


def _run_deterministic_route(df: pd.DataFrame, route: dict, question: str) -> dict:
    intent = route["intent"]
    if intent == "statistical_summary":
        insights = generate_statistical_insights(df)
        return {"insights": "\n".join(f"- {item}" for item in insights), "tool_result": None}

    if intent == "forecast":
        column = _select_numeric_column(df, route.get("column", ""))
        horizon = _forecast_horizon(question)
        method = str(route.get("method", "")).lower()
        if method == "moving_average":
            forecast = moving_average_forecast(df[column], horizon=horizon)
        elif method == "exponential_smoothing":
            forecast = exponential_smoothing_forecast(df[column], horizon=horizon)
        else:
            forecast = linear_trend_forecast(df[column], horizon=horizon)
        return {
            "insights": forecast.interpretation,
            "tool_result": {"type": "forecast", "column": column, "forecast": forecast},
        }

    if intent == "anomaly_detection":
        method = str(route.get("method", "")).lower()
        if method == "zscore":
            anomaly = detect_zscore(df)
        elif method == "iqr":
            anomaly = detect_iqr(df)
        else:
            anomaly = detect_isolation_forest(df)
        return {
            "insights": (
                f"{anomaly.method} found **{anomaly.n_anomalies:,} anomalous rows** "
                f"using: {', '.join(anomaly.columns_used)}."
            ),
            "tool_result": {"type": "anomaly", "anomaly": anomaly},
        }

    raise ValueError(f"Unsupported deterministic route: {intent}")


def _build_codegen_chain():
    if not LANGCHAIN_AVAILABLE:
        return None

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            ("user", "{user_prompt}"),
        ]
    )

    def _call_model(prompt_value):
        messages = prompt_value.to_messages()
        system_prompt = messages[0].content
        user_prompt = messages[1].content
        response, model = query_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=2048,
            temperature=0.3,
        )
        return {"response": response, "model": model}

    return prompt | RunnableLambda(_call_model)


def _generate_code(df: pd.DataFrame, question: str, history: list | None, max_tokens: int):
    prompt_text = build_context(df, question, history)

    if not LANGCHAIN_AVAILABLE:
        response, model_used = query_llm(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt_text,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response, model_used, False

    chain = _build_codegen_chain()
    payload = chain.invoke(
        {
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": prompt_text,
        }
    )
    return payload["response"], payload["model"], True


def run_query_langchain(
    df: pd.DataFrame,
    question: str,
    history: list | None = None,
    max_tokens: int = 2048,
) -> dict:
    """Run the dataframe question pipeline using LangChain for orchestration."""
    result = {
        "question": question,
        "llm_response": "",
        "code_generation_response": "",
        "code_blocks": [],
        "execution_results": [],
        "insights": "",
        "model_used": "",
        "error": None,
        "orchestrator": "langchain" if LANGCHAIN_AVAILABLE else "fallback",
    }

    response, model_used, _ = _generate_code(df, question, history, max_tokens)
    result["code_generation_response"] = response
    result["llm_response"] = ""
    result["model_used"] = model_used

    if response.startswith("❌"):
        result["error"] = response
        return result

    code_blocks = extract_python_code(response)
    result["code_blocks"] = code_blocks

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
        repair_note = (
            "\n\n## Repair Note\n"
            "A generated code block raised an execution error, so it was automatically corrected and re-run."
        )

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


def run_routed_query(
    question: str,
    df: pd.DataFrame | None = None,
    history: list | None = None,
    max_tokens: int = 2048,
    document_name: str | None = None,
) -> dict:
    """Route a question before either calling a trusted tool or generating code.

    Only the general dataframe-analysis route reaches the code generator. Forecast,
    anomaly, and statistical-summary routes call the project's deterministic modules.
    """
    is_document = bool(document_name and df is None)
    route = route_question(question, df=df, is_document=is_document)
    intent = route["intent"]

    result = {
        "question": question,
        "route": intent,
        "route_reason": route.get("reason", ""),
        "routing_source": route.get("routing_source", "heuristic"),
        "routing_model": route.get("routing_model", "none"),
        "code_blocks": [],
        "execution_results": [],
        "tool_result": None,
        "insights": "",
        "llm_response": "",
        "model_used": route.get("routing_model", "none"),
        "error": None,
        "orchestrator": "langchain-router" if LANGCHAIN_AVAILABLE else "fallback-router",
    }

    try:
        if intent == "document_qa":
            if not document_name:
                raise ValueError("Document questions require an active text document.")
            document_result = answer_document_question(question, document_name, max_tokens=max_tokens)
            result.update(
                insights=document_result["answer"],
                llm_response=document_result["answer"],
                tool_result={"type": "document", "sources": document_result.get("sources", [])},
                model_used=document_result.get("model_used", result["model_used"]),
                error=document_result.get("error"),
            )
            return result

        if df is None:
            raise ValueError("A tabular dataset is required for this question.")

        if intent in {"forecast", "anomaly_detection", "statistical_summary"}:
            deterministic_result = _run_deterministic_route(df, route, question)
            result.update(
                insights=deterministic_result["insights"],
                llm_response=deterministic_result["insights"],
                tool_result=deterministic_result["tool_result"],
            )
            return result

        code_result = run_query_langchain(df, question, history, max_tokens)
        code_result.update(
            route=intent,
            route_reason=result["route_reason"],
            routing_source=result["routing_source"],
            routing_model=result["routing_model"],
            orchestrator="langchain-router + " + code_result.get("orchestrator", "fallback"),
        )
        return code_result
    except Exception as exc:
        result["error"] = str(exc)
        return result

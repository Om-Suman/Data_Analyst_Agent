from __future__ import annotations

import io
import json
from datetime import datetime
from typing import Any, Optional
import pandas as pd
import plotly.graph_objects as go

from backend.session.state import SessionState
from backend.services.dataset_service import sanitize_dataframe_for_json
from modules.langchain_query import run_routed_query


def _serialize_execution_results(exec_results: list) -> list[dict[str, Any]]:
    serialized = []
    for res in exec_results:
        figures_json = []
        for fig in getattr(res, "figures", []):
            try:
                figures_json.append(json.loads(fig.to_json()))
            except Exception:
                pass

        dfs_json = {}
        for name, sub_df in getattr(res, "dataframes", {}).items():
            if isinstance(sub_df, pd.DataFrame):
                dfs_json[name] = sanitize_dataframe_for_json(sub_df.head(50))

        serialized.append({
            "code": getattr(res, "code", ""),
            "success": bool(getattr(res, "success", False)),
            "execution_time": round(float(getattr(res, "execution_time", 0.0)), 3),
            "stdout": str(getattr(res, "stdout", "")),
            "error": str(getattr(res, "error", "")) if getattr(res, "error", None) else None,
            "figures": figures_json,
            "dataframes": dfs_json,
        })
    return serialized


def _serialize_tool_result(tool_res: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not tool_res:
        return None
    res_type = tool_res.get("type")
    if res_type == "forecast":
        fc = tool_res.get("forecast")
        fig_spec = None
        if fc and hasattr(fc, "fig") and fc.fig is not None:
            try:
                fig_spec = json.loads(fc.fig.to_json())
            except Exception:
                pass
        return {
            "type": "forecast",
            "column": tool_res.get("column", ""),
            "figure_spec": fig_spec,
            "metrics": getattr(fc, "metrics", {}) if fc else {},
            "forecast_index": getattr(fc, "forecast_index", []) if fc else [],
            "forecast_values": getattr(fc, "forecast_values", []) if fc else [],
            "confidence_lower": getattr(fc, "confidence_lower", []) if fc else [],
            "confidence_upper": getattr(fc, "confidence_upper", []) if fc else [],
            "interpretation": getattr(fc, "interpretation", "") if fc else "",
        }
    elif res_type == "anomaly":
        an = tool_res.get("anomaly")
        fig_spec = None
        if an and hasattr(an, "fig") and an.fig is not None:
            try:
                fig_spec = json.loads(an.fig.to_json())
            except Exception:
                pass
        an_df = getattr(an, "anomaly_df", None)
        return {
            "type": "anomaly",
            "method": getattr(an, "method", "") if an else "",
            "n_anomalies": getattr(an, "n_anomalies", 0) if an else 0,
            "anomaly_indices": getattr(an, "anomaly_indices", []) if an else [],
            "anomalous_rows": sanitize_dataframe_for_json(an_df.head(50)) if isinstance(an_df, pd.DataFrame) else [],
            "columns_used": getattr(an, "columns_used", []) if an else [],
            "figure_spec": fig_spec,
        }
    elif res_type == "document":
        return {
            "type": "document",
            "sources": tool_res.get("sources", []),
        }
    return tool_res


def execute_ai_query(
    session: SessionState,
    question: str,
    max_tokens: Optional[int] = None,
    dataset_name: Optional[str] = None,
) -> dict[str, Any]:
    active_name = dataset_name or session.active_dataset
    if not active_name:
        raise ValueError("No active dataset or document loaded.")

    record = session.datasets.get(active_name)
    if not record:
        raise ValueError(f"Dataset '{active_name}' not found in session.")

    df = record.get("df")
    is_document = bool(record.get("text_content"))

    effective_max_tokens = max_tokens or session.config.get("max_tokens", 2048)

    raw_result = run_routed_query(
        question=question,
        df=df,
        history=session.query_history,
        max_tokens=effective_max_tokens,
        document_name=active_name if is_document else None,
    )

    code_blocks = raw_result.get("code_blocks", [])
    raw_exec_results = raw_result.get("execution_results", [])
    for idx, cb in enumerate(code_blocks):
        if idx < len(raw_exec_results):
            raw_exec_results[idx].code = cb

    exec_results = _serialize_execution_results(raw_exec_results)
    tool_result = _serialize_tool_result(raw_result.get("tool_result"))

    insights_text = raw_result.get("insights", "") or raw_result.get("llm_response", "")
    route = raw_result.get("route", "dataframe_analysis")
    route_reason = raw_result.get("route_reason", "")
    routing_source = raw_result.get("routing_source", "heuristic")
    model_used = raw_result.get("model_used", "")
    error = raw_result.get("error")

    # Add to query history if successful
    if not error:
        code_summary = "\n\n".join(code_blocks)
        session.add_query_to_history(
            question=question,
            code=code_summary,
            result_summary=insights_text[:500],
            dataset_name=active_name,
            route=route,
            model_used=model_used,
        )

    return {
        "question": question,
        "route": route,
        "route_reason": route_reason,
        "routing_source": routing_source,
        "model_used": model_used,
        "insights": insights_text,
        "code_blocks": code_blocks,
        "execution_results": exec_results,
        "tool_result": tool_result,
        "error": error,
    }


def get_query_history(session: SessionState) -> list[dict[str, Any]]:
    history = []
    for item in reversed(session.query_history):
        ts = item["timestamp"].isoformat() if hasattr(item["timestamp"], "isoformat") else str(item["timestamp"])
        history.append({
            "id": item["id"],
            "timestamp": ts,
            "question": item["question"],
            "code": item.get("code", ""),
            "result_summary": item.get("result_summary", ""),
            "dataset": item.get("dataset", ""),
            "route": item.get("route", "dataframe_analysis"),
            "model_used": item.get("model_used", ""),
        })
    return history


def export_query_history_csv(session: SessionState) -> bytes:
    if not session.query_history:
        df = pd.DataFrame(columns=["Timestamp", "Question", "Dataset", "Code", "Result Summary"])
    else:
        df = pd.DataFrame([
            {
                "Timestamp": e["timestamp"].isoformat() if hasattr(e["timestamp"], "isoformat") else str(e["timestamp"]),
                "Question": e["question"],
                "Dataset": e.get("dataset", ""),
                "Code": e.get("code", ""),
                "Result Summary": e.get("result_summary", ""),
            }
            for e in session.query_history
        ])
    return df.to_csv(index=False).encode("utf-8")


def execute_sql_query(session: SessionState, query: str, limit: int = 500) -> dict[str, Any]:
    import sqlite3
    import time
    import re

    df = session.get_active_df()
    if df is None or df.empty:
        return {
            "query": query,
            "success": False,
            "columns": [],
            "rows": [],
            "total_rows": 0,
            "execution_time": 0.0,
            "error": "No active dataset loaded in session. Please load a dataset first.",
        }

    q_clean = query.strip()
    if not q_clean:
        return {
            "query": query,
            "success": False,
            "columns": [],
            "rows": [],
            "total_rows": 0,
            "execution_time": 0.0,
            "error": "SQL query cannot be empty.",
        }

    # Restrict to safe SELECT operations only
    forbidden_tokens = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE", "REPLACE", "ATTACH", "DETACH", "PRAGMA", "EXEC"]
    normalized_q = re.sub(r"--.*", "", q_clean)
    normalized_q = re.sub(r"/\*.*?\*/", "", normalized_q, flags=re.DOTALL).upper()
    tokens = re.findall(r"\b[A-Z]+\b", normalized_q)

    for forbidden in forbidden_tokens:
        if forbidden in tokens:
            return {
                "query": query,
                "success": False,
                "columns": [],
                "rows": [],
                "total_rows": 0,
                "execution_time": 0.0,
                "error": f"Security restriction: '{forbidden}' statements are not permitted in SQL Studio. Only read-only SELECT queries are allowed.",
            }

    start_time = time.time()
    try:
        conn = sqlite3.connect(":memory:")
        # Register dataset under 'df' and 'data' and the dataset name sanitized
        df.to_sql("df", conn, index=False, if_exists="replace")
        df.to_sql("data", conn, index=False, if_exists="replace")
        if session.active_dataset:
            safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", session.active_dataset)
            if safe_name and safe_name not in ("df", "data"):
                df.to_sql(safe_name, conn, index=False, if_exists="replace")

        result_df = pd.read_sql_query(q_clean, conn)
        conn.close()
        exec_ms = round((time.time() - start_time) * 1000, 2)

        total_rows = len(result_df)
        limited_df = result_df.head(limit)
        sanitized_rows = sanitize_dataframe_for_json(limited_df)
        cols = list(result_df.columns)

        return {
            "query": query,
            "success": True,
            "columns": cols,
            "rows": sanitized_rows,
            "total_rows": total_rows,
            "execution_time": exec_ms,
            "error": None,
        }
    except Exception as e:
        exec_ms = round((time.time() - start_time) * 1000, 2)
        return {
            "query": query,
            "success": False,
            "columns": [],
            "rows": [],
            "total_rows": 0,
            "execution_time": exec_ms,
            "error": str(e),
        }

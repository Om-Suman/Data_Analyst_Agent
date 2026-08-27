from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    max_tokens: Optional[int] = None
    dataset_name: Optional[str] = None


class CodeExecutionResult(BaseModel):
    code: str
    success: bool
    execution_time: float
    stdout: str
    error: Optional[str] = None
    figures: list[dict[str, Any]] = []
    dataframes: dict[str, list[dict[str, Any]]] = {}


class QueryResponse(BaseModel):
    question: str
    route: str = "dataframe_analysis"
    route_reason: str = ""
    routing_source: str = "heuristic"
    model_used: str = ""
    insights: str = ""
    code_blocks: list[str] = []
    execution_results: list[CodeExecutionResult] = []
    tool_result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class QueryHistoryItem(BaseModel):
    id: str
    timestamp: str
    question: str
    code: str
    result_summary: str
    dataset: str
    route: Optional[str] = "dataframe_analysis"
    model_used: Optional[str] = ""


class QueryHistoryResponse(BaseModel):
    history: list[QueryHistoryItem]

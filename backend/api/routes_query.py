from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status
from backend.api.deps import get_current_session
from backend.session.state import SessionState
from backend.schemas.query import (
    QueryHistoryResponse,
    QueryRequest,
    QueryResponse,
    SQLQueryRequest,
    SQLQueryResponse,
)
from backend.services.query_service import (
    execute_ai_query,
    execute_sql_query,
    export_query_history_csv,
    get_query_history,
)

router = APIRouter(prefix="/query", tags=["AI Query"])


@router.post("", response_model=QueryResponse)
def ask_ai_query(
    req: QueryRequest,
    session: SessionState = Depends(get_current_session),
):
    try:
        res = execute_ai_query(
            session=session,
            question=req.question,
            max_tokens=req.max_tokens,
            dataset_name=req.dataset_name,
        )
        return res
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post("/sql", response_model=SQLQueryResponse)
def run_sql(
    req: SQLQueryRequest,
    session: SessionState = Depends(get_current_session),
):
    try:
        res = execute_sql_query(session=session, query=req.query, limit=req.limit or 500)
        return res
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get("/history", response_model=QueryHistoryResponse)
def list_history(session: SessionState = Depends(get_current_session)):
    return QueryHistoryResponse(history=get_query_history(session))


@router.delete("/history")
def clear_history(session: SessionState = Depends(get_current_session)):
    session.clear_query_history()
    return {"status": "ok", "message": "Query history cleared."}


@router.get("/history/export")
def export_history(session: SessionState = Depends(get_current_session)):
    csv_bytes = export_query_history_csv(session)
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="query_history.csv"'},
    )

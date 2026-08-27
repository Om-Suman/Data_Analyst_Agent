from __future__ import annotations

from typing import Any, Optional
from backend.session.state import SessionState
from modules.document_rag import (
    answer_document_question,
    get_document_bundle,
    build_document_bundle,
    store_document_bundle,
)


def query_document(
    session: SessionState,
    question: str,
    document_name: Optional[str] = None,
    max_tokens: Optional[int] = 1024,
) -> dict[str, Any]:
    active_name = document_name or session.active_dataset
    if not active_name:
        raise ValueError("No active document loaded.")

    record = session.datasets.get(active_name)
    if not record:
        raise ValueError(f"Document '{active_name}' not found.")

    text_content = record.get("text_content")
    if not text_content:
        raise ValueError(f"'{active_name}' is not a text document. Upload a TXT, PDF, DOCX, or OCR image first.")

    # Ensure document is indexed
    bundle = get_document_bundle(active_name, store=session.document_indexes)
    if not bundle:
        bundle = build_document_bundle(text_content, metadata={"name": active_name, **record.get("meta", {})})
        store_document_bundle(active_name, bundle, store=session.document_indexes)

    api_key = session.config.get("hf_api_key", "")
    primary = session.config.get("primary_model", "")
    fallback = session.config.get("fallback_model", "")

    result = answer_document_question(
        question=question,
        document_name=active_name,
        max_tokens=max_tokens or 1024,
        store=session.document_indexes,
        api_key=api_key,
        primary_model=primary,
        fallback_model=fallback,
    )

    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "engine": result.get("engine", "fallback"),
        "model_used": result.get("model_used"),
        "error": result.get("error"),
    }

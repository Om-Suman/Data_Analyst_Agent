from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel


class DocumentQARequest(BaseModel):
    question: str
    document_name: Optional[str] = None
    max_tokens: Optional[int] = 1024


class RetrievedSource(BaseModel):
    text: str
    score: float
    metadata: dict[str, Any] = {}


class DocumentQAResponse(BaseModel):
    answer: str
    sources: list[RetrievedSource] = []
    engine: str = "fallback"
    model_used: Optional[str] = None
    error: Optional[str] = None

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from backend.api.deps import get_current_session
from backend.session.state import SessionState
from backend.schemas.document import DocumentQARequest, DocumentQAResponse
from backend.services.document_service import query_document

router = APIRouter(prefix="/document", tags=["Document QA"])


@router.post("/qa", response_model=DocumentQAResponse)
def ask_document_qa(
    req: DocumentQARequest,
    session: SessionState = Depends(get_current_session),
):
    try:
        return query_document(
            session=session,
            question=req.question,
            document_name=req.document_name,
            max_tokens=req.max_tokens,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

from __future__ import annotations

from typing import Optional
from fastapi import Header, Request
from backend.session.state import SessionManager, SessionState


def get_current_session(
    request: Request,
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-ID"),
) -> SessionState:
    session_id = x_session_id
    if not session_id:
        session_id = request.cookies.get("session_id", "default")
    return SessionManager().get_session(session_id)

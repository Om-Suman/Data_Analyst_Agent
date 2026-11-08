from __future__ import annotations

import requests
from fastapi import APIRouter, Depends, HTTPException
from backend.api.deps import get_current_session
from backend.session.state import SessionState
from backend.schemas.config import ConfigResponse, ConfigUpdateRequest

router = APIRouter(tags=["Config & Settings"])


def _mask_api_key(key: str) -> str:
    key = (key or "").strip()
    if not key:
        return "Not configured"
    if len(key) <= 8:
        return "****"
    return f"{'*' * (len(key) - 4)}{key[-4:]}"


@router.get("/config", response_model=ConfigResponse)
def get_config(session: SessionState = Depends(get_current_session)):
    cfg = session.config
    key = cfg.get("groq_api_key", "")
    return ConfigResponse(
        has_api_key=bool(key and key.strip()),
        api_key_masked=_mask_api_key(key),
        primary_model=cfg.get("primary_model", "qwen/qwen3.8-27b"),
        fallback_model=cfg.get("fallback_model", "openai/gpt-oss-120b"),
        theme=cfg.get("theme", "light"),
        max_tokens=int(cfg.get("max_tokens", 2048)),
        temperature=float(cfg.get("temperature", 0.3)),
        default_outlier=cfg.get("default_outlier", "none"),
        default_missing_threshold=int(cfg.get("default_missing_threshold", 50)),
        auto_profile=bool(cfg.get("auto_profile", False)),
        auto_insights=bool(cfg.get("auto_insights", False)),
    )


@router.post("/config", response_model=ConfigResponse)
def update_config(
    req: ConfigUpdateRequest,
    session: SessionState = Depends(get_current_session),
):
    cfg = session.config
    if req.groq_api_key is not None and req.groq_api_key.strip():
        cfg["groq_api_key"] = req.groq_api_key.strip()
    if req.primary_model is not None and req.primary_model.strip():
        cfg["primary_model"] = req.primary_model.strip()
    if req.fallback_model is not None:
        cfg["fallback_model"] = req.fallback_model.strip()
    if req.theme is not None:
        cfg["theme"] = req.theme
    if req.max_tokens is not None:
        cfg["max_tokens"] = req.max_tokens
    if req.temperature is not None:
        cfg["temperature"] = req.temperature
    if req.default_outlier is not None:
        cfg["default_outlier"] = req.default_outlier
    if req.default_missing_threshold is not None:
        cfg["default_missing_threshold"] = req.default_missing_threshold
    if req.auto_profile is not None:
        cfg["auto_profile"] = req.auto_profile
    if req.auto_insights is not None:
        cfg["auto_insights"] = req.auto_insights

    return get_config(session)


@router.post("/session/reset")
def reset_session(session: SessionState = Depends(get_current_session)):
    session.reset()
    return {"status": "ok", "message": "Session reset successfully."}


@router.get("/config/connection")
def check_connection(session: SessionState = Depends(get_current_session)):
    key = session.config.get("groq_api_key", "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="Save an API key before testing the connection.")
    try:
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=8,
            allow_redirects=False,
        )
        if response.status_code in (401, 403):
            raise HTTPException(status_code=400, detail="Groq rejected this key or denied access. Check your API key and project permissions.")
        response.raise_for_status()
        models = {model["id"] for model in response.json()["data"] if model.get("active", True)}
    except (requests.RequestException, ValueError, KeyError, TypeError):
        raise HTTPException(status_code=502, detail="Could not verify the Groq connection. Try again shortly.")
    selected = {session.config.get("primary_model"), session.config.get("fallback_model")}
    return {"unavailable_models": sorted(model for model in selected if model and model not in models)}

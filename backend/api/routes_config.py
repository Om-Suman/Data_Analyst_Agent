from __future__ import annotations

from fastapi import APIRouter, Depends
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
    key = cfg.get("hf_api_key", "")
    return ConfigResponse(
        has_api_key=bool(key and key.strip()),
        api_key_masked=_mask_api_key(key),
        primary_model=cfg.get("primary_model", "deepseek-ai/DeepSeek-R1"),
        fallback_model=cfg.get("fallback_model", ""),
        theme=cfg.get("theme", "dark"),
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
    if req.hf_api_key is not None and req.hf_api_key.strip():
        cfg["hf_api_key"] = req.hf_api_key.strip()
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

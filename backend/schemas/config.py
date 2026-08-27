from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class ConfigResponse(BaseModel):
    has_api_key: bool = Field(description="Whether a valid API key is configured")
    api_key_masked: str = Field(description="Masked API key (e.g. *******1234)")
    primary_model: str = Field(default="deepseek-ai/DeepSeek-R1")
    fallback_model: str = Field(default="")
    theme: str = Field(default="dark")
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=0.3)
    default_outlier: str = Field(default="none")
    default_missing_threshold: int = Field(default=50)
    auto_profile: bool = Field(default=False)
    auto_insights: bool = Field(default=False)


class ConfigUpdateRequest(BaseModel):
    hf_api_key: Optional[str] = None
    primary_model: Optional[str] = None
    fallback_model: Optional[str] = None
    theme: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    default_outlier: Optional[str] = None
    default_missing_threshold: Optional[int] = None
    auto_profile: Optional[bool] = None
    auto_insights: Optional[bool] = None

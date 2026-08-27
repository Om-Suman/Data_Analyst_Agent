from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class StatisticalInsightsResponse(BaseModel):
    insights: list[str]


class AIInsightsRequest(BaseModel):
    max_tokens: Optional[int] = Field(default=1500)


class AIInsightsResponse(BaseModel):
    executive_summary: str = ""
    key_findings: list[str] = []
    trends: list[str] = []
    opportunities: list[str] = []
    risks: list[str] = []
    recommendations: list[str] = []
    data_story: str = ""
    cached: bool = False

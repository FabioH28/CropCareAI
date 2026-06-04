"""Advisor chat schemas."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class AdvisorChatRequest(BaseModel):
    message: str = Field(min_length=2, max_length=1200)
    context_mode: Literal["account", "current_scan"] = "account"
    model_mode: Literal["fast", "thinking"] = "fast"
    current_diagnosis: dict[str, Any] | None = None


class AdvisorContextImage(BaseModel):
    title: str
    url: str
    source: str


class AdvisorChatResponse(BaseModel):
    reply: str
    source: str
    llm_model: str | None = None
    context_summary: str
    context_images: list[AdvisorContextImage] = Field(default_factory=list)

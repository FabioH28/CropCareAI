"""Advisory service abstraction with local LLM and rules-based fallback support."""

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.enums import HealthStatus, SeverityLevel, UrgencyLevel
from app.schemas.diagnosis import AdvisoryPayload

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.plant.llm.explainer import (
    build_fallback_payload,
    build_prompt_context,
    build_response_schema,
    build_system_prompt,
    build_user_prompt,
    compose_advisory_text,
    normalize_advisory_payload,
)

settings = get_settings()
logger = get_logger(__name__)


class BaseAdvisoryService(ABC):
    """Abstract interface for future LLM-backed treatment advice generation."""

    @abstractmethod
    def generate(
        self,
        *,
        crop: str,
        disease: str | None,
        health_status: HealthStatus,
        confidence: float,
        severity: SeverityLevel,
        urgency: UrgencyLevel,
        farm_notes: str | None = None,
        analysis_payload: dict | None = None,
    ) -> AdvisoryPayload:
        """Return structured advisory guidance without changing the diagnosis."""


class RulesBasedAdvisoryService(BaseAdvisoryService):
    """Simple deterministic advisory generator used as a safe fallback."""

    def generate(
        self,
        *,
        crop: str,
        disease: str | None,
        health_status: HealthStatus,
        confidence: float,
        severity: SeverityLevel,
        urgency: UrgencyLevel,
        farm_notes: str | None = None,
        analysis_payload: dict | None = None,
    ) -> AdvisoryPayload:
        infected_area_percentage = None
        if analysis_payload:
            infected_area_percentage = analysis_payload.get("summary", {}).get("infected_area_percentage")

        payload = build_fallback_payload(
            crop=crop,
            disease=disease,
            health_status=health_status.value,
            confidence=confidence,
            severity=severity.value,
            urgency=urgency.value,
            infected_area_percentage=infected_area_percentage,
            farm_notes=farm_notes,
        )
        payload["advisory_source"] = "rules"
        payload["llm_model"] = None
        payload["advisory_text"] = compose_advisory_text(payload)
        return AdvisoryPayload.model_validate(payload)


class OllamaAdvisoryService(BaseAdvisoryService):
    """Local Ollama-backed explanation service with deterministic fallback."""

    def __init__(self, fallback_service: BaseAdvisoryService) -> None:
        self.fallback_service = fallback_service

    def generate(
        self,
        *,
        crop: str,
        disease: str | None,
        health_status: HealthStatus,
        confidence: float,
        severity: SeverityLevel,
        urgency: UrgencyLevel,
        farm_notes: str | None = None,
        analysis_payload: dict | None = None,
    ) -> AdvisoryPayload:
        fallback_payload = self.fallback_service.generate(
            crop=crop,
            disease=disease,
            health_status=health_status,
            confidence=confidence,
            severity=severity,
            urgency=urgency,
            farm_notes=farm_notes,
            analysis_payload=analysis_payload,
        )

        context = build_prompt_context(
            analysis_payload or {},
            crop=crop,
            disease=disease,
            health_status=health_status.value,
            confidence=confidence,
            severity=severity.value,
            urgency=urgency.value,
            farm_notes=farm_notes,
        )

        try:
            with httpx.Client(timeout=settings.llm_timeout_seconds) as client:
                response = client.post(
                    f"{settings.llm_base_url.rstrip('/')}/api/generate",
                    json={
                        "model": settings.llm_model,
                        "system": build_system_prompt(),
                        "prompt": build_user_prompt(context),
                        "format": build_response_schema(),
                        "stream": False,
                        "keep_alive": "10m",
                        "options": {
                            "temperature": 0.2,
                            "num_predict": 340,
                        },
                    },
                )
                response.raise_for_status()

            payload = response.json()
            content = payload.get("response", "").strip()
            parsed = json.loads(content) if content else {}
            normalized = normalize_advisory_payload(parsed, fallback_payload.model_dump())
            normalized["advisory_source"] = "ollama"
            normalized["llm_model"] = settings.llm_model
            normalized["advisory_text"] = compose_advisory_text(normalized)
            return AdvisoryPayload.model_validate(normalized)
        except Exception as exc:
            logger.warning("ollama_advisory_failed", extra={"reason": str(exc), "model": settings.llm_model})
            fallback = fallback_payload.model_copy()
            fallback.advisory_text = f"{fallback.advisory_text}\n\nNote: Local LLM explanation was unavailable, so a fallback explanation was used."
            return fallback


@lru_cache
def get_advisory_service() -> BaseAdvisoryService:
    rules_service = RulesBasedAdvisoryService()
    if settings.enable_llm_advisory and settings.llm_provider.lower() == "ollama":
        return OllamaAdvisoryService(fallback_service=rules_service)
    return rules_service

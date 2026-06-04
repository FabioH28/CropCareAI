import json

from app.models.enums import HealthStatus, SeverityLevel, UrgencyLevel
from app.services.advisory_service import OllamaAdvisoryService, RulesBasedAdvisoryService


def test_rules_advisory_service_returns_structured_payload():
    service = RulesBasedAdvisoryService()

    result = service.generate(
        crop="tomato",
        disease="early_blight",
        health_status=HealthStatus.diseased,
        confidence=0.92,
        severity=SeverityLevel.high,
        urgency=UrgencyLevel.urgent,
        farm_notes="Humidity is high this week.",
    )

    assert "tomato" in result.advisory_text.lower()
    assert len(result.treatment_steps) >= 3
    assert any("humid" in tip.lower() for tip in result.prevention_tips)
    assert result.what_happened
    assert result.likely_cause
    assert len(result.action_explanations) >= 2


def test_ollama_advisory_service_parses_structured_response(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "response": json.dumps(
                    {
                        "what_happened": "The tomato leaf shows a small diseased area.",
                        "likely_cause": "It may be early blight.",
                        "severity_summary": "Severity looks medium and needs attention soon.",
                        "why_it_matters": "It can spread and reduce healthy leaf area.",
                        "treatment_steps": [
                            "Inspect nearby leaves.",
                            "Remove clearly damaged leaf tissue.",
                            "Use a crop-safe treatment matched to the disease.",
                        ],
                        "action_explanations": [
                            "This helps you check whether it is spreading.",
                            "This can reduce further spread.",
                            "This makes the treatment more effective.",
                        ],
                        "prevention_tips": [
                            "Improve airflow around the crop.",
                            "Avoid overhead watering.",
                            "Clean tools between plants.",
                        ],
                        "urgency_guidance": "Act within the next day or two.",
                        "follow_up_recommendation": "Re-check the plant after 48 hours.",
                    }
                )
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("app.services.advisory_service.httpx.Client", FakeClient)

    service = OllamaAdvisoryService(fallback_service=RulesBasedAdvisoryService())
    result = service.generate(
        crop="tomato",
        disease="early_blight",
        health_status=HealthStatus.diseased,
        confidence=0.91,
        severity=SeverityLevel.high,
        urgency=UrgencyLevel.urgent,
        analysis_payload={"summary": {"infected_area_percentage": 12.5}},
    )

    assert result.advisory_source == "ollama"
    assert result.llm_model is not None
    assert "What happened:" in result.advisory_text
    assert result.treatment_steps[0] == "Inspect nearby leaves."

"""Helpers for turning plant CV/DL outputs into LLM-friendly context."""

from __future__ import annotations

import json
from typing import Any


def is_unknown_crop(crop: str | None) -> bool:
    normalized = (crop or "").strip().lower()
    return normalized in {"", "unknown", "unknown_leaf_crop", "unsupported_crop"}


def build_prompt_context(
    analysis_payload: dict[str, Any],
    *,
    crop: str,
    disease: str | None,
    health_status: str,
    confidence: float,
    severity: str,
    urgency: str,
    farm_notes: str | None = None,
) -> dict[str, Any]:
    summary = analysis_payload.get("summary", {})
    comparison = analysis_payload.get("classification", {}).get("comparison", {})
    segmentation = analysis_payload.get("segmentation", {})
    outputs = analysis_payload.get("outputs", [])

    return {
        "crop": crop or summary.get("predicted_crop"),
        "health_status": health_status or summary.get("health_status"),
        "predicted_disease": disease or summary.get("predicted_disease"),
        "confidence_score": confidence or summary.get("confidence_score"),
        "severity_level": severity or summary.get("severity_level"),
        "urgency_level": urgency or summary.get("urgency_level"),
        "infected_area_percentage": summary.get("infected_area_percentage"),
        "leaf_area_percentage": segmentation.get("leaf_area_percentage"),
        "baseline_vs_improved": comparison,
        "farm_notes": farm_notes,
        "key_visuals": [
            {
                "stage": item.get("stage"),
                "label": item.get("label"),
            }
            for item in outputs[:6]
        ],
    }


def build_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "what_happened": {"type": "string"},
            "likely_cause": {"type": "string"},
            "severity_summary": {"type": "string"},
            "why_it_matters": {"type": "string"},
            "treatment_steps": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 3,
            },
            "action_explanations": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 3,
            },
            "prevention_tips": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 3,
            },
            "urgency_guidance": {"type": "string"},
            "follow_up_recommendation": {"type": "string"},
        },
        "required": [
            "what_happened",
            "likely_cause",
            "severity_summary",
            "why_it_matters",
            "treatment_steps",
            "action_explanations",
            "prevention_tips",
            "urgency_guidance",
            "follow_up_recommendation",
        ],
    }


def build_system_prompt() -> str:
    return (
        "You explain plant diagnosis results in very short, clear, human language. "
        "Use the structured computer-vision output faithfully. "
        "Do not overclaim. If the diagnosis is uncertain, say it may be caused by the most likely issue. "
        "Keep treatment steps practical and explain why each step helps. "
        "Return valid JSON only."
    )


def build_user_prompt(context: dict[str, Any]) -> str:
    return (
        "Create a short plant diagnosis explanation from this structured analysis.\n"
        "Requirements:\n"
        "- Keep each field short and plain.\n"
        "- Each string should be one short sentence when possible.\n"
        "- Mention what happened, the likely cause, severity, what to do now, and why the actions help.\n"
        "- Be specific about urgency and infected area when available.\n"
        "- Do not number the treatment steps.\n"
        "- Use at most 3 treatment steps, 3 action explanations, and 3 prevention tips.\n"
        "- Avoid long explanations and filler words.\n"
        "- Do not mention that you are an AI model.\n"
        "- If crop is unknown_leaf_crop, say the crop is outside the supported set and do not invent a disease name.\n"
        "- Return JSON that matches the required schema.\n\n"
        f"Analysis context:\n{json.dumps(context, indent=2)}"
    )


def build_fallback_payload(
    *,
    crop: str,
    disease: str | None,
    health_status: str,
    confidence: float,
    severity: str,
    urgency: str,
    infected_area_percentage: float | None = None,
    farm_notes: str | None = None,
) -> dict[str, Any]:
    if is_unknown_crop(crop):
        affected_text = (
            f" Visible abnormal area is about {infected_area_percentage:.2f}%."
            if infected_area_percentage is not None
            else ""
        )
        what_happened = (
            "The leaf does not confidently match one of the crops supported by this model."
            f"{affected_text}"
        )
        likely_cause = "This is likely an unsupported crop or an image outside the training set."
        if health_status == "healthy":
            treatment_steps = [
                "Retake one close photo with a single leaf filling most of the frame.",
                "Confirm the crop manually before trusting any treatment choice.",
            ]
            action_explanations = [
                "A tighter crop gives the vision pipeline a cleaner leaf view.",
                "Manual crop confirmation prevents acting on a wrong crop label.",
            ]
        else:
            treatment_steps = [
                "Retake one clear close photo of the affected leaf.",
                "Inspect nearby leaves manually for the same pattern.",
                "Do not rely on a named disease label until this crop is added to training.",
            ]
            action_explanations = [
                "A better image may still improve the stress estimate.",
                "Manual checking tells you whether the issue is spreading.",
                "Unsupported crops should not be forced into the wrong treatment plan.",
            ]
        prevention_tips = [
            "Keep the leaf well lit and centered when capturing the next image.",
            "Separate clearly stressed leaves from healthy ones during inspection.",
            "Extend the training set if this crop needs reliable automated diagnosis.",
        ]
        return {
            "what_happened": what_happened,
            "likely_cause": likely_cause,
            "severity_summary": f"The visible stress looks {severity} and needs {urgency} attention.",
            "why_it_matters": "A wrong crop identity can lead to the wrong disease name and the wrong treatment choice.",
            "treatment_steps": treatment_steps,
            "action_explanations": action_explanations,
            "prevention_tips": prevention_tips,
            "urgency_guidance": f"Urgency is {urgency}. Act based on the visible symptoms, not on a guessed crop name.",
            "follow_up_recommendation": "Re-scan after taking a clearer close leaf photo or after extending the model to this crop.",
        }

    disease_name = disease or "a stress pattern without a precise disease match"
    infected_text = (
        f" Around {infected_area_percentage:.2f}% of the visible leaf area looks affected."
        if infected_area_percentage is not None
        else ""
    )
    what_happened = (
        f"The {crop} leaf appears {health_status} with about {confidence:.0%} confidence.{infected_text}"
    )
    likely_cause = f"The most likely cause is {disease_name}."
    if farm_notes and "humid" in farm_notes.lower():
        likely_cause += " Humid conditions can make this kind of problem spread more easily."

    if health_status == "healthy":
        treatment_steps = [
            "Keep monitoring the plant over the next few days.",
            "Maintain balanced watering and airflow.",
        ]
        action_explanations = [
            "Regular checks help catch new symptoms early.",
            "Stable growing conditions reduce avoidable stress.",
        ]
    else:
        treatment_steps = [
            "Check nearby leaves for similar symptoms.",
            "Remove or isolate clearly affected leaf material if it is safe to do so.",
            "Use a crop-safe treatment only after matching it to the likely issue.",
        ]
        action_explanations = [
            "Nearby inspection helps you see whether the issue is spreading.",
            "Removing infected tissue can reduce further spread.",
            "Using the right treatment avoids wasting time or stressing the plant more.",
        ]
    prevention_tips = [
        "Avoid overwatering or overhead watering when disease spread is a concern.",
        "Improve spacing and airflow around the crop canopy where possible.",
        "Keep tools and hands clean between affected and healthy plants.",
    ]
    if farm_notes and "humid" in farm_notes.lower():
        prevention_tips[-1] = "During humid weather, inspect the crop more often because symptoms can spread faster."

    return {
        "what_happened": what_happened,
        "likely_cause": likely_cause,
        "severity_summary": f"This looks {severity} and needs {urgency} attention.",
        "why_it_matters": "Acting early helps limit spread and protect healthy leaf area.",
        "treatment_steps": treatment_steps,
        "action_explanations": action_explanations[: len(treatment_steps)],
        "prevention_tips": prevention_tips[:3],
        "urgency_guidance": f"Urgency is {urgency}. Act sooner if symptoms spread quickly.",
        "follow_up_recommendation": "Re-check the plant within 48 to 72 hours.",
    }


def normalize_advisory_payload(payload: dict[str, Any], fallback_payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(fallback_payload)
    for key, value in payload.items():
        if isinstance(value, str) and value.strip():
            normalized[key] = value.strip()
        elif isinstance(value, list) and value:
            normalized[key] = [str(item).strip() for item in value if str(item).strip()]

    step_count = len(normalized.get("treatment_steps", []))
    explanations = normalized.get("action_explanations", [])
    if len(explanations) < step_count:
        fallback_explanations = fallback_payload.get("action_explanations", [])
        explanations = explanations + fallback_explanations[len(explanations) : step_count]
        normalized["action_explanations"] = explanations[:step_count]
    return normalized


def compose_advisory_text(payload: dict[str, Any]) -> str:
    parts = [
        f"What happened: {payload['what_happened']}",
        f"Likely cause: {payload['likely_cause']}",
        f"Severity: {payload['severity_summary']}",
        f"Why it matters: {payload['why_it_matters']}",
    ]
    if payload.get("treatment_steps"):
        numbered_steps = " ".join(
            f"{index + 1}. {step}"
            for index, step in enumerate(payload["treatment_steps"])
        )
        parts.append(f"What to do now: {numbered_steps}")
    return "\n".join(parts)


def render_farmer_friendly_summary(analysis_payload: dict[str, Any]) -> str:
    summary = analysis_payload.get("summary", {})
    crop = summary.get("predicted_crop", "plant")
    health_status = summary.get("health_status", "unknown")
    disease = summary.get("predicted_disease") or "no specific disease pattern"
    confidence = summary.get("confidence_score", 0.0)
    infected_area = summary.get("infected_area_percentage", 0.0)
    severity = summary.get("severity_level", "unknown")

    if is_unknown_crop(crop):
        return (
            "The analysis suggests this leaf is outside the supported crop set. "
            f"Visible condition: {health_status}. "
            f"Confidence in the rejection step: {confidence:.2f}. "
            f"Estimated affected area: {infected_area:.2f}%. "
            f"Severity: {severity}."
        )

    return (
        f"The analysis suggests the {crop} leaf is {health_status}. "
        f"Most likely pattern: {disease}. "
        f"Confidence: {confidence:.2f}. "
        f"Estimated infected area: {infected_area:.2f}%. "
        f"Severity: {severity}."
    )

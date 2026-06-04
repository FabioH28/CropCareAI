"""Short plant-advisor replies backed by saved account context."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date
from typing import Any, Literal

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

FAST_ADVISOR_MODEL = "llama3.2:3b"
THINKING_ADVISOR_MODEL = "llama3:8b"


class AdvisorChatService:
    def __init__(self, db_client) -> None:
        self.db = db_client

    def ask(
        self,
        *,
        user_id: str,
        message: str,
        context_mode: Literal["account", "current_scan"] = "account",
        model_mode: Literal["fast", "thinking"] = "fast",
        current_diagnosis: Mapping[str, Any] | None = None,
    ) -> dict[str, object]:
        context = self._load_context(
            user_id,
            context_mode=context_mode,
            current_diagnosis=current_diagnosis,
        )
        context_summary = self._build_context_summary(context)
        context_images = self._build_context_images(context)
        selected_model = self._model_for_mode(model_mode)
        question = message.strip().lower()

        if self._is_greeting(question):
            return {
                "reply": self._build_greeting_reply(context),
                "source": "advisor",
                "llm_model": None,
                "context_summary": context_summary,
                "context_images": context_images,
            }

        fallback_reply = self._build_fallback_reply(message, context)
        is_overview_question = self._is_overview_question(message.strip().lower())

        if is_overview_question:
            return {
                "reply": fallback_reply,
                "source": "rules",
                "llm_model": None,
                "context_summary": context_summary,
                "context_images": context_images,
            }

        if settings.enable_llm_advisory and settings.llm_provider.lower() == "ollama":
            llm_reply = self._ask_ollama(message=message, context=context, model=selected_model, model_mode=model_mode)
            if llm_reply:
                cleaned_reply = self._clean_reply(llm_reply)
                if self._is_bad_llm_reply(llm_reply) or self._is_bad_llm_reply(cleaned_reply):
                    cleaned_reply = fallback_reply
                return {
                    "reply": cleaned_reply,
                    "source": "ollama",
                    "llm_model": selected_model,
                    "context_summary": context_summary,
                    "context_images": context_images,
                }

        return {
            "reply": fallback_reply,
            "source": "rules",
            "llm_model": None,
            "context_summary": context_summary,
            "context_images": context_images,
        }

    def _load_context(
        self,
        user_id: str,
        *,
        context_mode: Literal["account", "current_scan"],
        current_diagnosis: Mapping[str, Any] | None,
    ) -> dict[str, object]:
        profile_rows = self.db.table("profiles").select("*").eq("id", user_id).limit(1).execute().data or []
        plants = self.db.table("plants").select("*").eq("user_id", user_id).neq("status", "archived").execute().data or []
        diagnoses = self.db.table("diagnoses").select("*").eq("user_id", user_id).order("created_at", desc=True).execute().data or []
        treatment_logs = self.db.table("treatment_logs").select("*").eq("user_id", user_id).order("applied_at", desc=True).execute().data or []

        today = date.today().isoformat()
        follow_ups_due = [
            item
            for item in treatment_logs
            if item.get("follow_up_date") and item["follow_up_date"] <= today and item.get("status") != "completed"
        ]

        latest_diagnosis = diagnoses[0] if diagnoses else None
        latest_advisory = latest_diagnosis.get("advisory_payload") if latest_diagnosis else None

        return {
            "profile": profile_rows[0] if profile_rows else {},
            "plants": plants,
            "diagnoses": diagnoses,
            "treatment_logs": treatment_logs,
            "latest_diagnosis": latest_diagnosis,
            "latest_advisory": latest_advisory if isinstance(latest_advisory, Mapping) else {},
            "follow_ups_due": follow_ups_due,
            "context_mode": context_mode,
            "current_diagnosis": dict(current_diagnosis or {}) if current_diagnosis else {},
        }

    def _build_context_summary(self, context: Mapping[str, object]) -> str:
        plants = context.get("plants") or []
        diagnoses = context.get("diagnoses") or []
        follow_ups_due = context.get("follow_ups_due") or []
        current_diagnosis = context.get("current_diagnosis")
        current_text = "current scan, " if isinstance(current_diagnosis, Mapping) and current_diagnosis else ""
        return f"Using {current_text}{len(plants)} plants, full {len(diagnoses)} diagnosis history item(s), and {len(follow_ups_due)} due follow-ups."

    def _build_fallback_reply(self, message: str, context: Mapping[str, object]) -> str:
        current_diagnosis = context.get("current_diagnosis")
        latest_diagnosis = current_diagnosis if isinstance(current_diagnosis, Mapping) and current_diagnosis else context.get("latest_diagnosis")
        latest_advisory = (
            latest_diagnosis.get("advisory")
            if isinstance(latest_diagnosis, Mapping) and isinstance(latest_diagnosis.get("advisory"), Mapping)
            else context.get("latest_advisory")
        )
        question = message.strip().lower()

        if self._is_overview_question(question):
            return self._build_overview_reply(context)

        if isinstance(latest_diagnosis, Mapping) and latest_diagnosis:
            crop = str(latest_diagnosis.get("predicted_crop") or "plant").replace("_", " ").title()
            disease = str(latest_diagnosis.get("predicted_disease") or latest_diagnosis.get("health_status") or "latest result")
            disease = disease.replace("_", " ").replace("__", " ").title()
            confidence = float(latest_diagnosis.get("confidence_score") or 0.0)
            severity = str(latest_diagnosis.get("severity_level") or "unknown").replace("_", " ")
            urgency = str(latest_diagnosis.get("urgency_level") or "unknown").replace("_", " ")
            what_happened = self._advisory_value(latest_advisory, "what_happened")
            likely_cause = self._advisory_value(latest_advisory, "likely_cause")
            why_it_matters = self._advisory_value(latest_advisory, "why_it_matters")
            severity_summary = self._advisory_value(latest_advisory, "severity_summary")
            urgency_guidance = self._advisory_value(latest_advisory, "urgency_guidance")
            follow_up = self._advisory_value(latest_advisory, "follow_up_recommendation")
            treatment_steps = self._advisory_list(latest_advisory, "treatment_steps")
            first_steps = " ".join(treatment_steps[:2])

            if any(keyword in question for keyword in ["cause", "why", "reason"]):
                return f"{likely_cause} {why_it_matters}".strip()
            if any(keyword in question for keyword in ["severity", "degree", "bad", "serious", "urgent"]):
                return f"{severity_summary} {urgency_guidance}".strip()
            if any(keyword in question for keyword in ["do", "treat", "fix", "action", "next"]):
                return f"Next step: {first_steps} {follow_up}".strip()
            if any(keyword in question for keyword in ["what", "result", "happened", "scan"]):
                return what_happened

            return (
                f"Latest result: {crop} looks most like {disease} at {confidence:.0%} confidence. "
                f"Severity is {severity} and urgency is {urgency}. Next: {first_steps}"
            ).strip()

        follow_ups_due = context.get("follow_ups_due") or []
        plants = context.get("plants") or []
        if follow_ups_due:
            next_due = follow_ups_due[0]
            return (
                f"You do not have a recent diagnosis to explain yet, but you have {len(follow_ups_due)} follow-up task(s) due. "
                f"The next one is '{next_due.get('treatment_action', 'treatment follow-up')}'. "
                "Finish that check and save a fresh scan so I can give more specific advice."
            )
        if plants:
            return (
                f"You have {len(plants)} plants in CropCare AI. Save a diagnosis from the Diagnose page and I will explain "
                "what happened, what likely caused it, and what to do next in plain language."
            )
        return (
            "Start by adding a plant and running a diagnosis. Once a scan is saved, I can explain the result, the likely cause, "
            "and the next treatment steps for that plant."
        )

    def _ask_ollama(
        self,
        *,
        message: str,
        context: Mapping[str, object],
        model: str,
        model_mode: Literal["fast", "thinking"],
    ) -> str | None:
        general_care_question = self._is_general_care_question(message.strip().lower())
        private_memory = (
            "This is a general plant-care question. Do not use saved scan history unless the farmer explicitly asks about their saved plants, scans, diagnoses, or history."
            if general_care_question
            else self._build_private_memory(context)
        )
        system_prompt = (
            "You are CropCare AI's plant expert assistant. Act like a practical agronomist. "
            "The app gives you private memory from the user's account. Use it silently. "
            "Never mention JSON, objects, schemas, keys, fields, payloads, context, or provided data. "
            "Never use markdown bold, asterisks, code blocks, or technical data-format language. "
            "For account-specific questions, do not invent diseases, certainty, treatments, dates, or plants that are not in the private memory. "
            "For general plant-care questions, answer from practical gardening knowledge and do not mention saved scan history unless the farmer asks for it. "
            "For tree planting, never tell the user to plant deeper than the root ball. Say to keep the root flare visible and any graft union above the soil. "
            "If the farmer asks about plants/history, summarize all relevant crops and scans, not only the latest scan. "
            "If there are no saved plant profiles, say that plainly and use saved diagnosis scans as history. "
            "Answer directly without greetings."
        )
        prompt = (
            "Answer the farmer using the private app memory below. "
            "If the question is about the user's saved plants or scans, use the current scan first when present, then saved plant profiles, full diagnosis history, and due follow-ups. "
            "If the question is general plant care, answer the question normally and do not use saved scan history. "
            "If model confidence is low or evidence is uncertain, say so plainly. "
            "Explain what the result likely means, what to check visually, and the next safe action. "
            "Keep the answer under 120 words. "
            f"Mode: {model_mode}. In fast mode, be brief and decisive. In thinking mode, reason a little more carefully. "
            "If images are listed, say the app shows them as thumbnails, but do not claim you inspected pixels. "
            "Use plain language, no tables.\n\n"
            f"Farmer question:\n{message}\n\n"
            f"Private app memory for you only:\n{private_memory}"
        )

        try:
            with httpx.Client(timeout=settings.llm_timeout_seconds) as client:
                response = client.post(
                    f"{settings.llm_base_url.rstrip('/')}/api/generate",
                    json={
                        "model": model,
                        "system": system_prompt,
                        "stream": False,
                        "keep_alive": "10m",
                        "options": {"temperature": 0.2, "num_predict": 130 if model_mode == "fast" else 240},
                        "prompt": prompt,
                    },
                )
                response.raise_for_status()
            reply = str(response.json().get("response") or "").strip()
            return reply or None
        except Exception as exc:
            logger.warning("advisor_chat_ollama_failed", extra={"reason": str(exc), "model": model})
            return None

    def _is_overview_question(self, question: str) -> bool:
        normalized = re.sub(r"[^a-z\s]", " ", question)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        how_to_phrases = [
            "how to plant",
            "how do i plant",
            "how should i plant",
            "how can i plant",
            "how to grow",
            "how do i grow",
            "how should i grow",
            "how can i grow",
        ]
        if any(phrase in normalized for phrase in how_to_phrases):
            return False

        overview_phrases = [
            "my plants",
            "my plant history",
            "my saved",
            "saved plants",
            "saved scans",
            "scan history",
            "diagnosis history",
            "plant profiles",
            "all plants",
            "all scans",
            "all diagnoses",
        ]
        overview_verbs = ["show", "list", "overview", "summary", "summarize", "review", "advice", "advise"]
        return any(phrase in normalized for phrase in overview_phrases) and any(
            term in normalized for term in overview_verbs
        )

    def _is_general_care_question(self, question: str) -> bool:
        normalized = re.sub(r"[^a-z\s]", " ", question)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        care_phrases = [
            "how to plant",
            "how do i plant",
            "how should i plant",
            "how can i plant",
            "how to grow",
            "how do i grow",
            "how should i grow",
            "how can i grow",
            "how to care",
            "how do i care",
            "how should i care",
        ]
        account_terms = ["my plant", "my plants", "my scan", "my scans", "my diagnosis", "my diagnoses", "my history"]
        return any(phrase in normalized for phrase in care_phrases) and not any(
            term in normalized for term in account_terms
        )

    def _is_greeting(self, question: str) -> bool:
        normalized = re.sub(r"[^a-z\s]", " ", question).strip()
        normalized = re.sub(r"\s+", " ", normalized)
        greeting_phrases = {
            "hi",
            "hello",
            "hey",
            "yo",
            "sup",
            "good morning",
            "good afternoon",
            "good evening",
        }
        return normalized in greeting_phrases or normalized.startswith(("hi ", "hello ", "hey "))

    def _build_greeting_reply(self, context: Mapping[str, object]) -> str:
        name = self._user_first_name(context)
        greeting = f"Hi {name}" if name else "Hi"
        return (
            f"{greeting}, what do you want to check today? "
            "You can ask about your latest scan, your plant history, treatment steps, or upload a new leaf photo."
        )

    def _user_first_name(self, context: Mapping[str, object]) -> str:
        profile = context.get("profile")
        if not isinstance(profile, Mapping):
            return ""
        raw_name = str(profile.get("full_name") or "").strip()
        if not raw_name:
            return ""
        return raw_name.split()[0]

    def _build_overview_reply(self, context: Mapping[str, object]) -> str:
        plants = context.get("plants") or []
        diagnoses = context.get("diagnoses") or []
        follow_ups_due = context.get("follow_ups_due") or []

        lines: list[str] = []
        if isinstance(plants, list) and plants:
            crop_names = [
                str(plant.get("custom_name") or plant.get("crop_type") or "Unnamed plant").replace("_", " ").title()
                for plant in plants
                if isinstance(plant, Mapping)
            ]
            lines.append(f"Saved plant profiles: {', '.join(crop_names[:8])}.")
        else:
            lines.append("No saved plant profiles yet.")

        if isinstance(diagnoses, list) and diagnoses:
            diagnosis_parts = [self._format_diagnosis_summary(item) for item in diagnoses if isinstance(item, Mapping)]
            lines.append(f"Saved scan history: {'; '.join(part for part in diagnosis_parts if part)}.")
        else:
            lines.append("No saved scans yet.")

        if isinstance(follow_ups_due, list) and follow_ups_due:
            lines.append(f"You have {len(follow_ups_due)} follow-up task(s) due.")

        lines.append(
            "Advice: create plant profiles for the crops you scan, then prioritize any diseased or medium/high urgency result for inspection, isolation of badly affected leaves, and a fresh follow-up scan."
        )
        return " ".join(line for line in lines if line).strip()

    def _format_diagnosis_summary(self, diagnosis: Mapping[str, object]) -> str:
        crop = str(diagnosis.get("predicted_crop") or "plant").replace("_", " ").title()
        disease = str(diagnosis.get("predicted_disease") or diagnosis.get("health_status") or "unknown result")
        disease = disease.replace("__", " ").replace("_", " ").title()
        confidence = diagnosis.get("confidence_score")
        confidence_text = f", {float(confidence):.0%} confidence" if isinstance(confidence, int | float) else ""
        severity = str(diagnosis.get("severity_level") or "").replace("_", " ")
        severity_text = f", {severity} severity" if severity else ""
        date_text = str(diagnosis.get("created_at") or "")[:10]
        date_suffix = f" on {date_text}" if date_text else ""
        return f"{crop}: {disease}{confidence_text}{severity_text}{date_suffix}"

    def _build_private_memory(self, context: Mapping[str, object]) -> str:
        plants = context.get("plants") or []
        diagnoses = context.get("diagnoses") or []
        follow_ups_due = context.get("follow_ups_due") or []
        current_diagnosis = context.get("current_diagnosis")
        images = self._build_context_images(context)

        sections: list[str] = []
        if isinstance(current_diagnosis, Mapping) and current_diagnosis:
            sections.append(f"Current scan: {self._format_diagnosis_summary(current_diagnosis)}.")

        if isinstance(plants, list) and plants:
            plant_lines = []
            for plant in plants:
                if isinstance(plant, Mapping):
                    name = str(plant.get("custom_name") or plant.get("crop_type") or "Unnamed plant").replace("_", " ").title()
                    status = str(plant.get("status") or "unknown")
                    location = str(plant.get("zone_or_field") or "").strip()
                    plant_lines.append(f"{name} ({status}{f', {location}' if location else ''})")
            sections.append(f"Saved plant profiles: {', '.join(plant_lines)}.")
        else:
            sections.append("Saved plant profiles: none.")

        if isinstance(diagnoses, list) and diagnoses:
            scan_lines = [self._format_diagnosis_summary(item) for item in diagnoses if isinstance(item, Mapping)]
            sections.append("Full saved scan history: " + "; ".join(scan_lines) + ".")
        else:
            sections.append("Full saved scan history: none.")

        if isinstance(follow_ups_due, list) and follow_ups_due:
            sections.append(f"Due follow-ups: {len(follow_ups_due)}.")
        else:
            sections.append("Due follow-ups: none.")

        if images:
            image_titles = [image["title"] for image in images[:8]]
            sections.append("Images shown in the app: " + "; ".join(image_titles) + ".")

        return "\n".join(sections)

    def _clean_reply(self, reply: str) -> str:
        cleaned = reply.strip()
        cleaned = cleaned.replace("**", "")
        cleaned = re.sub(r"(?i)\bthis is a json object\b[:.]?\s*", "", cleaned)
        cleaned = re.sub(r"(?i)\bjson object\b", "plant record", cleaned)
        cleaned = re.sub(r"(?i)\bjson\b", "app data", cleaned)
        cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _is_bad_llm_reply(self, reply: str) -> bool:
        lowered = reply.lower()
        bad_patterns = [
            r"\bjson\b",
            r"\bobject\b",
            r"\bpayload\b",
            r"\bschema\b",
            r"\bfield\b",
            r"provided context",
            r"account context",
            r"compact memory",
            r"private app memory",
        ]
        return any(re.search(pattern, lowered) for pattern in bad_patterns) or "**" in reply

    def _model_for_mode(self, model_mode: Literal["fast", "thinking"]) -> str:
        if model_mode == "thinking":
            return THINKING_ADVISOR_MODEL or settings.llm_model
        return FAST_ADVISOR_MODEL

    def _summarize_records(self, records: object) -> list[dict[str, object]]:
        if not isinstance(records, list):
            return []

        allowed_keys = {
            "id",
            "plant_id",
            "custom_name",
            "crop_type",
            "status",
            "zone_or_field",
            "notes",
            "predicted_crop",
            "predicted_disease",
            "health_status",
            "confidence_score",
            "severity_level",
            "urgency_level",
            "image_url",
            "created_at",
            "treatment_action",
            "follow_up_date",
            "applied_at",
        }
        summarized: list[dict[str, object]] = []
        for record in records:
            if isinstance(record, Mapping):
                summarized.append({key: value for key, value in record.items() if key in allowed_keys})
        return summarized

    def _build_context_images(self, context: Mapping[str, object]) -> list[dict[str, str]]:
        images: list[dict[str, str]] = []
        seen: set[str] = set()

        def add_image(title: str, url: object, source: str) -> None:
            if not isinstance(url, str) or not url.strip() or url in seen:
                return
            seen.add(url)
            images.append({"title": title, "url": url, "source": source})

        current_diagnosis = context.get("current_diagnosis")
        if isinstance(current_diagnosis, Mapping):
            crop = str(current_diagnosis.get("predicted_crop") or "Current scan").replace("_", " ").title()
            add_image(f"{crop} current scan", current_diagnosis.get("image_url"), "current_scan")
            analysis_images = current_diagnosis.get("analysis_images")
            if isinstance(analysis_images, list):
                for item in analysis_images:
                    if isinstance(item, Mapping):
                        add_image(str(item.get("title") or "Analysis image"), item.get("url"), "current_scan_analysis")

        diagnoses = context.get("diagnoses")
        if isinstance(diagnoses, list):
            for index, diagnosis in enumerate(diagnoses, start=1):
                if not isinstance(diagnosis, Mapping):
                    continue
                crop = str(diagnosis.get("predicted_crop") or "Plant").replace("_", " ").title()
                date_text = str(diagnosis.get("created_at") or f"history {index}")[:10]
                add_image(f"{crop} diagnosis - {date_text}", diagnosis.get("image_url"), "diagnosis_history")

        return images

    def _advisory_value(self, advisory: object, key: str) -> str:
        if isinstance(advisory, Mapping):
            value = advisory.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _advisory_list(self, advisory: object, key: str) -> list[str]:
        if isinstance(advisory, Mapping):
            value = advisory.get(key)
            if isinstance(value, list):
                return [str(item).strip() for item in value if str(item).strip()]
        return []

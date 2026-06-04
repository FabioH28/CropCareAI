"""Inference service abstraction with CV analysis and DL checkpoint integration."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from app.core.config import get_settings
from app.core.exceptions import AppException
from app.core.logging import get_logger
from app.models.enums import HealthStatus, SeverityLevel, UrgencyLevel
from app.schemas.diagnosis import InferencePayload
from app.utils.files import sanitize_filename

settings = get_settings()
logger = get_logger(__name__)
ProgressCallback = Callable[[str, float, str], None]


# Shared normalization helpers for the handoff between CV output and DL output.
def _coerce_health_status(value: str | None) -> HealthStatus:
    normalized = (value or "").strip().lower()
    if normalized == HealthStatus.healthy.value:
        return HealthStatus.healthy
    if normalized in {HealthStatus.suspicious.value, "uncertain"}:
        return HealthStatus.suspicious
    return HealthStatus.diseased


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _label_metadata(label: str) -> dict[str, Any]:
    crop, _, disease = label.partition("__")
    normalized_disease = disease or "unknown"
    is_healthy = normalized_disease == "healthy"
    return {
        "crop": crop or "unknown",
        "disease": normalized_disease,
        "is_healthy": is_healthy,
        "health_status": "healthy" if is_healthy else "diseased",
    }


def _prediction_is_known_crop(prediction: dict[str, Any] | None) -> bool:
    if not prediction:
        return False
    return bool(prediction.get("is_known_crop", True))


def _gate_supported_similarity(prediction: dict[str, Any] | None) -> float:
    gate = (prediction or {}).get("open_set_gate") or {}
    return _as_float(gate.get("top_supported_similarity"), 0.0)


def _gate_unknown_similarity(prediction: dict[str, Any] | None) -> float:
    gate = (prediction or {}).get("open_set_gate") or {}
    if "top_unknown_similarity" not in gate or gate.get("top_unknown_similarity") is None:
        return float("-inf")
    return _as_float(gate.get("top_unknown_similarity"), float("-inf"))


def _gate_margin(prediction: dict[str, Any] | None) -> float:
    return _gate_supported_similarity(prediction) - _gate_unknown_similarity(prediction)


def _has_strong_known_crop_support(prediction: dict[str, Any] | None) -> bool:
    if not _prediction_is_known_crop(prediction):
        return False

    gate = (prediction or {}).get("open_set_gate") or {}
    if not gate:
        return _as_float((prediction or {}).get("confidence"), 0.0) >= settings.dl_confidence_threshold + 0.1

    minimum_supported = _as_float(gate.get("minimum_supported_similarity"), 0.58)
    unknown_margin = _as_float(gate.get("unknown_margin"), 0.03)
    supported_similarity = _gate_supported_similarity(prediction)
    margin = _gate_margin(prediction)
    return supported_similarity >= max(minimum_supported + 0.05, 0.68) and margin >= max(unknown_margin + 0.04, 0.07)


def _combine_open_set_gate_details(predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    branch_gates: dict[str, Any] = {}
    known_branches: list[str] = []
    unknown_branches: list[str] = []

    for branch_name, prediction in predictions.items():
        if _prediction_is_known_crop(prediction):
            known_branches.append(branch_name)
        else:
            unknown_branches.append(branch_name)
        branch_gates[branch_name] = prediction.get("open_set_gate")

    return {
        "branch_gates": branch_gates,
        "known_branches": known_branches,
        "unknown_branches": unknown_branches,
        "is_known_crop": not unknown_branches,
    }


def _build_unknown_crop_prediction(
    predictions: dict[str, dict[str, Any]],
    *,
    branch_used: str,
    fusion_strategy: str,
) -> dict[str, Any]:
    primary_prediction = predictions.get("leaf_roi") or predictions.get("full_image") or next(iter(predictions.values()))
    gate_details = _combine_open_set_gate_details(predictions)
    return {
        "predicted_label": "unknown_leaf_crop",
        "crop": "unknown_leaf_crop",
        "disease": None,
        "is_healthy": False,
        "health_status": "uncertain",
        "confidence": _as_float(primary_prediction.get("confidence"), 0.0),
        "model_name": primary_prediction.get("model_name"),
        "branch_used": branch_used,
        "fusion_strategy": fusion_strategy,
        "branches": predictions,
        "class_probabilities": primary_prediction.get("class_probabilities") or {},
        "crop_probabilities": primary_prediction.get("crop_probabilities") or {},
        "is_known_crop": False,
        "matched_supported_label": primary_prediction.get("matched_supported_label"),
        "matched_supported_crop": primary_prediction.get("matched_supported_crop"),
        "open_set_gate": gate_details,
    }


# When both image branches are available, keep the fusion logic in one place so
# the rest of the service only has to deal with one "best" DL answer.
def fuse_dl_predictions(
    predictions: dict[str, dict[str, Any]],
    analysis: dict[str, Any],
) -> dict[str, Any] | None:
    if not predictions:
        return None

    if len(predictions) == 1:
        branch_name, branch_prediction = next(iter(predictions.items()))
        fused = dict(branch_prediction)
        fused["branch_used"] = branch_name
        fused["fusion_strategy"] = "single_branch_only"
        fused["branches"] = predictions
        return fused

    leaf_prediction = predictions.get("leaf_roi")
    full_prediction = predictions.get("full_image")
    if not leaf_prediction:
        fused = dict(full_prediction or next(iter(predictions.values())))
        fused["branch_used"] = "full_image"
        fused["fusion_strategy"] = "full_image_only"
        fused["branches"] = predictions
        return fused

    if not full_prediction:
        fused = dict(leaf_prediction)
        fused["branch_used"] = "leaf_roi"
        fused["fusion_strategy"] = "leaf_roi_only"
        fused["branches"] = predictions
        return fused

    roi_info = analysis.get("roi") or {}
    roi_fill_ratio = _as_float(roi_info.get("roi_fill_percentage"), 0.0) / 100.0
    leaf_confidence = _as_float(leaf_prediction.get("confidence"), 0.0)
    full_confidence = _as_float(full_prediction.get("confidence"), 0.0)
    cv_infected_area = _as_float((analysis.get("summary") or {}).get("infected_area_percentage"), 0.0)
    leaf_probs = leaf_prediction.get("class_probabilities") or {}
    full_probs = full_prediction.get("class_probabilities") or {}
    open_set_gate = _combine_open_set_gate_details(predictions)
    leaf_known = _prediction_is_known_crop(leaf_prediction)
    full_known = _prediction_is_known_crop(full_prediction)

    # If the crop-support gate rejects the image, do not let probability fusion
    # reintroduce a supported crop label afterward.
    if not leaf_known and not full_known:
        return _build_unknown_crop_prediction(
            predictions,
            branch_used="open_set_rejection",
            fusion_strategy="all_branches_rejected_as_unsupported",
        )

    if leaf_known != full_known:
        if leaf_known and roi_fill_ratio >= 0.18 and _has_strong_known_crop_support(leaf_prediction):
            fused = dict(leaf_prediction)
            fused["branch_used"] = "leaf_roi"
            fused["fusion_strategy"] = "leaf_roi_open_set_override"
            fused["branches"] = predictions
            fused["open_set_gate"] = open_set_gate
            return fused

        if full_known and _has_strong_known_crop_support(full_prediction):
            fused = dict(full_prediction)
            fused["branch_used"] = "full_image"
            fused["fusion_strategy"] = "full_image_open_set_override"
            fused["branches"] = predictions
            fused["open_set_gate"] = open_set_gate
            return fused

        return _build_unknown_crop_prediction(
            predictions,
            branch_used="open_set_rejection",
            fusion_strategy="mixed_branch_predictions_rejected_as_unsupported",
        )

    if leaf_probs and full_probs:
        roi_weight = 0.65 if roi_fill_ratio >= 0.18 else 0.55
        full_weight = 1.0 - roi_weight
        labels = set(full_probs) | set(leaf_probs)
        fused_probabilities = {
            label: full_weight * _as_float(full_probs.get(label), 0.0) + roi_weight * _as_float(leaf_probs.get(label), 0.0)
            for label in labels
        }
        fused_label = max(fused_probabilities, key=fused_probabilities.get)
        fused_confidence = _as_float(fused_probabilities[fused_label], 0.0)
        metadata = _label_metadata(fused_label)
        # Preserve cascade metadata when both branches refined through the
        # same specialist — the fused softmax is already the specialist's
        # output in that case, so the cascade trail belongs on the fused
        # result too.
        leaf_cascade = bool(leaf_prediction.get("cascade_used"))
        full_cascade = bool(full_prediction.get("cascade_used"))
        any_cascade = leaf_cascade or full_cascade
        cascade_source = leaf_prediction if leaf_cascade else (full_prediction if full_cascade else None)
        cascade_route = cascade_source.get("cascade_route") if cascade_source else None
        if leaf_cascade and full_cascade and leaf_prediction.get("cascade_route") != full_prediction.get("cascade_route"):
            cascade_route = "mixed_branch_cascade_routes"
        fused = {
            "predicted_label": fused_label,
            "crop": metadata["crop"],
            "disease": metadata["disease"],
            "is_healthy": metadata["is_healthy"],
            "health_status": "healthy"
            if metadata["is_healthy"]
            else ("uncertain" if fused_confidence < settings.dl_confidence_threshold else "diseased"),
            "confidence": fused_confidence,
            "model_name": leaf_prediction.get("model_name") or full_prediction.get("model_name"),
            "branch_used": "probability_fusion",
            "fusion_strategy": "weighted_probability_fusion_leaf_roi",
            "branches": predictions,
            "class_probabilities": fused_probabilities,
            "is_known_crop": True,
            "matched_supported_label": fused_label,
            "matched_supported_crop": metadata["crop"],
            "open_set_gate": open_set_gate,
            "cascade_used": any_cascade,
            "cascade_route": cascade_route,
            "unified_prediction": (cascade_source or {}).get("unified_prediction"),
            "specialist_prediction": (cascade_source or {}).get("specialist_prediction"),
            "ensemble_prediction": (cascade_source or {}).get("ensemble_prediction"),
        }
        return fused

    if leaf_prediction.get("predicted_label") == full_prediction.get("predicted_label"):
        preferred = leaf_prediction if leaf_confidence >= full_confidence - 0.02 else full_prediction
        strategy = "branch_agreement_prefer_leaf_roi"
    elif roi_fill_ratio >= 0.22 and leaf_confidence >= full_confidence - 0.05:
        preferred = leaf_prediction
        strategy = "prefer_leaf_roi_on_disagreement"
    elif full_confidence >= leaf_confidence + 0.10:
        preferred = full_prediction
        strategy = "prefer_full_image_high_margin"
    else:
        preferred = leaf_prediction
        strategy = "default_leaf_roi_priority"

    alternate = full_prediction if preferred is leaf_prediction else leaf_prediction
    if (
        preferred.get("is_healthy")
        and not alternate.get("is_healthy")
        and cv_infected_area >= 4.0
        and _as_float(alternate.get("confidence"), 0.0) >= _as_float(preferred.get("confidence"), 0.0) - 0.05
    ):
        preferred = alternate
        strategy = "cv_override_prefers_diseased_branch"

    fused = dict(preferred)
    fused["branch_used"] = "leaf_roi" if preferred is leaf_prediction else "full_image"
    fused["fusion_strategy"] = strategy
    fused["branches"] = predictions
    fused["open_set_gate"] = open_set_gate
    return fused


def merge_plant_analysis(
    analysis: dict[str, Any],
    dl_prediction: dict[str, Any] | None,
    *,
    default_model_version: str,
) -> tuple[dict[str, Any], str]:
    # Start from the CV summary, then let the trained classifier override the
    # label when it has a confident, usable answer.
    summary = dict(analysis.get("summary") or {})
    cv_health_status = _coerce_health_status(summary.get("health_status"))
    cv_confidence = _as_float(summary.get("confidence_score"), 0.0)
    cv_infected_area = round(_as_float(summary.get("infected_area_percentage"), 0.0), 2)

    final_summary = {
        "predicted_crop": summary.get("predicted_crop") or "unknown_leaf_crop",
        "health_status": cv_health_status.value,
        "predicted_disease": summary.get("predicted_disease"),
        "confidence_score": round(cv_confidence, 3),
        "severity_level": summary.get("severity_level") or SeverityLevel.medium.value,
        "urgency_level": summary.get("urgency_level") or UrgencyLevel.medium.value,
        "infected_area_percentage": cv_infected_area,
    }

    classifier_source = "cv_heuristics"
    decision_strategy = "cv_only"
    model_version = str(analysis.get("pipeline", default_model_version))

    if dl_prediction:
        dl_health_status = _coerce_health_status(dl_prediction.get("health_status"))
        dl_confidence = _as_float(dl_prediction.get("confidence"), cv_confidence)
        dl_model_name = str(dl_prediction.get("model_name") or "dl-classifier")
        dl_known_crop = bool(dl_prediction.get("is_known_crop", True))

        if not dl_known_crop:
            final_summary.update(
                {
                    "predicted_crop": "unknown_leaf_crop",
                    "predicted_disease": None,
                    "confidence_score": round(cv_confidence, 3),
                    "health_status": cv_health_status.value,
                }
            )
            classifier_source = "open_set_crop_gate"
            decision_strategy = "unsupported_crop_rejected_keep_cv_summary"
            model_version = f"{analysis.get('pipeline', default_model_version)}+{dl_model_name}+open_set_gate"
        else:
            final_summary.update(
                {
                    "predicted_crop": dl_prediction.get("crop") or final_summary["predicted_crop"],
                    "predicted_disease": None if dl_prediction.get("is_healthy") else dl_prediction.get("disease"),
                    "confidence_score": round(dl_confidence, 3),
                    "health_status": dl_health_status.value,
                }
            )

            classifier_source = "efficientnet_checkpoint"
            decision_strategy = "dl_label_cv_severity"
            model_version = f"{analysis.get('pipeline', default_model_version)}+{dl_model_name}"

            # The DL cascade is the disease authority. Only hedge a DL "healthy"
            # toward "suspicious" when the STRICT CV verdict strongly disagrees
            # (says diseased) over a substantial area — so the sensitive CV
            # highlight alone never flips a healthy verdict (disease vs highlight
            # are decoupled: DL/strict-CV decide, the highlight only visualizes).
            if dl_health_status == HealthStatus.healthy and cv_health_status == HealthStatus.diseased and cv_infected_area >= 15.0:
                final_summary["health_status"] = HealthStatus.suspicious.value
                final_summary["predicted_disease"] = None
                final_summary["confidence_score"] = round(min(dl_confidence, max(cv_confidence, 0.79)), 3)
                decision_strategy = "dl_cv_disagreement_marked_suspicious"

    classification = analysis.setdefault("classification", {})
    classification["final"] = final_summary
    analysis["summary"] = final_summary

    model_integration = analysis.setdefault("model_integration", {})
    model_integration.update(
        {
            "classifier_source": classifier_source,
            "severity_source": "cv_consensus_segmentation",
            "decision_strategy": decision_strategy,
            "final_model_version": model_version,
        }
    )

    if dl_prediction:
        dl_section = model_integration.setdefault("dl_classifier", {})
        dl_section.update(
            {
                "used": True,
                "predicted_label": dl_prediction.get("predicted_label"),
                "crop": dl_prediction.get("crop"),
                "disease": dl_prediction.get("disease"),
                "is_healthy": bool(dl_prediction.get("is_healthy")),
                "health_status": _coerce_health_status(dl_prediction.get("health_status")).value,
                "confidence": round(_as_float(dl_prediction.get("confidence"), 0.0), 4),
                "model_name": dl_prediction.get("model_name"),
                "branch_used": dl_prediction.get("branch_used"),
                "fusion_strategy": dl_prediction.get("fusion_strategy"),
                "branch_predictions": dl_prediction.get("branches"),
                "is_known_crop": bool(dl_prediction.get("is_known_crop", True)),
                "matched_supported_label": dl_prediction.get("matched_supported_label"),
                "matched_supported_crop": dl_prediction.get("matched_supported_crop"),
                "open_set_gate": dl_prediction.get("open_set_gate"),
                "cascade_used": bool(dl_prediction.get("cascade_used", False)),
                "cascade_route": dl_prediction.get("cascade_route"),
                "unified_prediction": dl_prediction.get("unified_prediction"),
                "specialist_prediction": dl_prediction.get("specialist_prediction"),
                "ensemble_prediction": dl_prediction.get("ensemble_prediction"),
            }
        )
        classification["deep_learning"] = dict(dl_section)

    return analysis, model_version


class BaseInferenceService(ABC):
    """Abstract interface for pluggable crop and disease inference engines."""

    @abstractmethod
    async def predict(
        self,
        image_bytes: bytes,
        filename: str,
        content_type: str,
        *,
        stored_image_path: str | None = None,
        crop_hint: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> InferencePayload:
        """Return structured predictions for a plant image."""


class MockInferenceService(BaseInferenceService):
    """Deterministic placeholder used when the real CV pipeline is unavailable."""

    async def predict(
        self,
        image_bytes: bytes,
        filename: str,
        content_type: str,
        *,
        stored_image_path: str | None = None,
        crop_hint: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> InferencePayload:
        self._report_progress(progress_callback, "running_mock_inference", 65.0, "Running fallback classifier")
        extension = Path(filename).suffix.lower()
        predicted_crop = crop_hint or ("tomato" if extension in {".jpg", ".jpeg"} else "potato")
        health_status = HealthStatus.diseased if len(image_bytes) % 2 else HealthStatus.suspicious
        predicted_disease = "early_blight" if predicted_crop == "tomato" else "late_blight"
        confidence = 0.91 if health_status == HealthStatus.diseased else 0.78
        severity = SeverityLevel.high if health_status == HealthStatus.diseased else SeverityLevel.medium
        urgency = UrgencyLevel.urgent if health_status == HealthStatus.diseased else UrgencyLevel.medium

        return InferencePayload(
            predicted_crop=predicted_crop,
            health_status=health_status,
            predicted_disease=predicted_disease,
            confidence_score=confidence,
            severity_level=severity,
            urgency_level=urgency,
            model_version="mock-cropcare-v1",
            raw_prediction_json={
                "pipeline": "mock",
                "filename": filename,
                "content_type": content_type,
                "byte_size": len(image_bytes),
                "stored_image_path": stored_image_path,
                "crop_hint": crop_hint,
                "top_prediction": predicted_disease,
                "next_integration": "Switch ENABLE_MOCK_INFERENCE=false to use the full plant CV pipeline.",
            },
        )

    @staticmethod
    def _report_progress(
        progress_callback: ProgressCallback | None,
        stage_key: str,
        progress_percent: float,
        stage_label: str,
    ) -> None:
        if progress_callback:
            progress_callback(stage_key, progress_percent, stage_label)


class PlantVisionInferenceService(BaseInferenceService):
    """Runs CV analysis plus the trained plant classifier in the ML environment."""

    def __init__(self, fallback_service: BaseInferenceService | None = None) -> None:
        self.project_root = Path(__file__).resolve().parents[3]
        self.ml_root = self.project_root / "ml" / "plant"
        self.cv_script_path = self.ml_root / "cv" / "analyze.py"
        self.dl_script_path = self.ml_root / "dl" / "infer.py"
        self.cascade_script_path = self.ml_root / "dl" / "cascade.py"
        self.default_checkpoint_path = self.ml_root / "artifacts" / "production" / "best_model.pt"
        self.legacy_checkpoint_path = self.ml_root / "artifacts" / "checkpoints" / "best_model.pt"
        self.python_path = self._resolve_python_path()
        self.dl_checkpoint_path = self._resolve_checkpoint_path()
        self.specialist_paths = self._resolve_specialist_paths()
        # Kept for backward compatibility with code that referenced the apple-only field.
        self.apple_specialist_path = self.specialist_paths.get("apple")
        self.uploads_root = Path(settings.uploads_root).resolve()
        self.uploads_root.mkdir(parents=True, exist_ok=True)
        self.fallback_service = fallback_service

    async def predict(
        self,
        image_bytes: bytes,
        filename: str,
        content_type: str,
        *,
        stored_image_path: str | None = None,
        crop_hint: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> InferencePayload:
        try:
            # Each request gets its own analysis folder so the JSON and preview
            # images stay grouped together.
            self._report_progress(progress_callback, "preparing_workspace", 12.0, "Preparing analysis workspace")
            run_id = uuid4().hex
            output_dir = self.uploads_root / "analysis-runs" / run_id
            output_dir.mkdir(parents=True, exist_ok=True)
            output_json = output_dir / "analysis.json"
            input_path = self._resolve_input_path(
                image_bytes=image_bytes,
                filename=filename,
                stored_image_path=stored_image_path,
                output_dir=output_dir,
            )

            self._report_progress(progress_callback, "running_cv_pipeline", 22.0, "Running leaf isolation and CV analysis")
            analysis = await self._run_analysis(
                input_path=input_path,
                output_dir=output_dir,
                output_json=output_json,
                crop_hint=crop_hint,
            )
            self._report_progress(progress_callback, "cv_pipeline_complete", 58.0, "Computer vision analysis finished")
            model_integration = analysis.setdefault("model_integration", {})
            model_integration.setdefault(
                "cv_pipeline",
                {
                    "used": True,
                    "pipeline": analysis.get("pipeline", settings.default_model_version),
                },
            )
            dl_section = model_integration.setdefault(
                "dl_classifier",
                {
                    "enabled": settings.enable_dl_classifier,
                    "checkpoint_path": str(self.dl_checkpoint_path),
                    "checkpoint_exists": self.dl_checkpoint_path.exists(),
                    "used": False,
                },
            )

            dl_prediction: dict[str, Any] | None = None
            if settings.enable_dl_classifier and self.dl_checkpoint_path.exists():
                try:
                    # Run both the full image and the isolated leaf when possible,
                    # then fuse them into one DL decision.
                    self._report_progress(progress_callback, "running_dl_classifier", 68.0, "Running deep-learning classifier")
                    branch_predictions = await self._run_branch_predictions(
                        analysis=analysis,
                        output_dir=output_dir,
                        input_path=input_path,
                        progress_callback=progress_callback,
                    )
                    dl_prediction = fuse_dl_predictions(branch_predictions, analysis)
                except Exception as dl_exc:
                    logger.warning(
                        "dl_classifier_inference_failed",
                        extra={"reason": str(dl_exc), "image_name": filename},
                    )
                    dl_section["error"] = str(dl_exc)
                else:
                    dl_section["used"] = True
                    self._report_progress(progress_callback, "dl_classifier_complete", 88.0, "Deep-learning classification finished")
            elif settings.enable_dl_classifier:
                dl_section["error"] = f"Checkpoint not found at {self.dl_checkpoint_path}"

            self._report_progress(progress_callback, "merging_results", 91.0, "Combining CV and classifier results")
            analysis, model_version = merge_plant_analysis(
                analysis,
                dl_prediction,
                default_model_version=settings.default_model_version,
            )
            summary = analysis["summary"]

            return InferencePayload(
                predicted_crop=summary["predicted_crop"],
                health_status=_coerce_health_status(summary["health_status"]),
                predicted_disease=summary.get("predicted_disease"),
                confidence_score=float(summary["confidence_score"]),
                severity_level=SeverityLevel(summary["severity_level"]),
                urgency_level=UrgencyLevel(summary["urgency_level"]),
                model_version=model_version,
                raw_prediction_json=analysis,
            )
        except Exception as exc:
            # If the full plant stack fails, fall back to the deterministic mock
            # path instead of leaving the UI with no answer at all.
            logger.warning("plant_vision_inference_failed", extra={"reason": str(exc), "image_name": filename})
            if not self.fallback_service:
                raise AppException(
                    message="Plant CV inference failed.",
                    code="plant_cv_inference_failed",
                    details={"reason": str(exc)},
                ) from exc

            fallback = await self.fallback_service.predict(
                image_bytes=image_bytes,
                filename=filename,
                content_type=content_type,
                stored_image_path=stored_image_path,
                crop_hint=crop_hint,
            )
            fallback.raw_prediction_json["fallback_reason"] = str(exc)
            fallback.raw_prediction_json["fallback_from"] = "plant_vision_subprocess"
            return fallback

    def _resolve_python_path(self) -> Path:
        candidates = [
            self.ml_root / ".venv" / "Scripts" / "python.exe",
            self.ml_root / ".venv" / "bin" / "python",
            Path(sys.executable),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError("No Python executable available for the plant ML environment.")

    def _resolve_checkpoint_path(self) -> Path:
        configured_path = (settings.dl_checkpoint_path or "").strip()
        if not configured_path:
            if self.default_checkpoint_path.exists():
                return self.default_checkpoint_path
            if self.legacy_checkpoint_path.exists():
                return self.legacy_checkpoint_path
            return self.default_checkpoint_path

        candidate = Path(configured_path)
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        return candidate.resolve()

    def _resolve_apple_specialist_path(self) -> Path | None:
        return self._resolve_one_specialist_path(settings.dl_apple_specialist_path)

    def _resolve_one_specialist_path(self, configured_path: str | None) -> Path | None:
        configured_path = (configured_path or "").strip()
        if not configured_path:
            return None
        candidate = Path(configured_path)
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        resolved = candidate.resolve()
        return resolved if resolved.exists() else None

    def _resolve_specialist_paths(self) -> dict[str, Path]:
        crop_configs = {
            "apple": settings.dl_apple_specialist_path,
            "tomato": settings.dl_tomato_specialist_path,
            "grape": settings.dl_grape_specialist_path,
            "corn": settings.dl_corn_specialist_path,
            "pepper": settings.dl_pepper_specialist_path,
            "potato": settings.dl_potato_specialist_path,
        }
        return {
            crop: path
            for crop, raw in crop_configs.items()
            if (path := self._resolve_one_specialist_path(raw)) is not None
        }

    async def _run_analysis(
        self,
        *,
        input_path: Path,
        output_dir: Path,
        output_json: Path,
        crop_hint: str | None,
    ) -> dict[str, Any]:
        # The CV pipeline runs in the ML environment as a subprocess so the API
        # layer stays thin and doesn't have to own the image-processing runtime.
        command = [
            str(self.python_path),
            str(self.cv_script_path),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-json",
            str(output_json),
        ]
        if crop_hint:
            command.extend(["--crop-hint", crop_hint])

        process = await asyncio.to_thread(
            subprocess.run,
            command,
            cwd=str(self.ml_root),
            capture_output=True,
            env={**os.environ, "LOKY_MAX_CPU_COUNT": "1"},
        )

        if process.returncode != 0:
            raise AppException(
                message="Plant CV subprocess failed.",
                code="plant_cv_subprocess_failed",
                details={
                    "return_code": process.returncode,
                    "stderr": process.stderr.decode("utf-8", errors="ignore").strip(),
                },
            )

        payload_text = process.stdout.decode("utf-8", errors="ignore").strip()
        if payload_text:
            analysis = json.loads(payload_text)
        elif output_json.exists():
            analysis = json.loads(output_json.read_text(encoding="utf-8"))
        else:
            raise AppException(
                message="Plant CV subprocess returned no JSON output.",
                code="plant_cv_output_missing",
            )

        return self._annotate_analysis_paths(analysis, input_path=input_path, output_dir=output_dir)

    async def _run_branch_predictions(
        self,
        *,
        analysis: dict[str, Any],
        output_dir: Path,
        input_path: Path,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, dict[str, Any]]:
        # These branch names line up with what the frontend later shows as the
        # source of the classifier decision.
        branch_inputs = self._resolve_dl_input_paths(analysis, output_dir=output_dir, input_path=input_path)
        predictions: dict[str, dict[str, Any]] = {}
        total_branches = max(len(branch_inputs), 1)
        for index, (branch_name, branch_path) in enumerate(branch_inputs.items(), start=1):
            branch_progress = 68.0 + (index - 1) * (16.0 / total_branches)
            branch_label = "Running leaf ROI classifier" if branch_name == "leaf_roi" else "Running whole-image classifier"
            self._report_progress(progress_callback, f"classifying_{branch_name}", branch_progress, branch_label)
            branch_prediction = await self._run_dl_inference(input_path=branch_path)
            branch_prediction["input_path"] = str(branch_path)
            branch_prediction["branch_name"] = branch_name
            predictions[branch_name] = branch_prediction
        return predictions

    async def _run_dl_inference(self, *, input_path: Path) -> dict[str, Any]:
        # Route through the cascade entrypoint when at least one crop specialist
        # is configured — the cascade JSON is a superset of infer.py output, so
        # downstream code works either way.
        if self.specialist_paths:
            command = [
                str(self.python_path),
                str(self.cascade_script_path),
                str(input_path),
                "--unified",
                str(self.dl_checkpoint_path),
                "--threshold",
                str(settings.dl_confidence_threshold),
                "--routing-threshold",
                str(settings.dl_cascade_routing_threshold),
                "--tta-passes",
                str(settings.dl_tta_passes),
            ]
            for crop, path in self.specialist_paths.items():
                command.extend(["--specialist", f"{crop}={path}"])
        else:
            command = [
                str(self.python_path),
                str(self.dl_script_path),
                str(input_path),
                "--checkpoint",
                str(self.dl_checkpoint_path),
                "--threshold",
                str(settings.dl_confidence_threshold),
                "--tta-passes",
                str(settings.dl_tta_passes),
            ]

        process = await asyncio.to_thread(
            subprocess.run,
            command,
            cwd=str(self.ml_root),
            capture_output=True,
            env={**os.environ, "LOKY_MAX_CPU_COUNT": "1"},
        )

        if process.returncode != 0:
            raise AppException(
                message="Deep learning classifier subprocess failed.",
                code="dl_classifier_subprocess_failed",
                details={
                    "return_code": process.returncode,
                    "stderr": process.stderr.decode("utf-8", errors="ignore").strip(),
                },
            )

        payload_text = process.stdout.decode("utf-8", errors="ignore").strip()
        if not payload_text:
            raise AppException(
                message="Deep learning classifier returned no JSON output.",
                code="dl_classifier_output_missing",
            )
        return json.loads(payload_text)

    def _resolve_dl_input_paths(
        self,
        analysis: dict[str, Any],
        *,
        output_dir: Path,
        input_path: Path,
    ) -> dict[str, Path]:
        # Default to the full image, then add the leaf-only branch if the CV pass
        # already exported a clean ROI crop.
        branch_inputs = {"full_image": input_path}
        roi_info = analysis.get("roi") or {}
        relative_path = roi_info.get("classifier_primary_relative_path")
        if roi_info.get("leaf_detected") and relative_path:
            roi_candidate = output_dir / str(relative_path)
            if roi_candidate.exists():
                branch_inputs["leaf_roi"] = roi_candidate
        return branch_inputs

    def _resolve_input_path(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        stored_image_path: str | None,
        output_dir: Path,
    ) -> Path:
        # Reuse the saved upload when we already have one; otherwise write a local
        # working copy beside the rest of the analysis artifacts.
        if stored_image_path:
            resolved = self.uploads_root / Path(stored_image_path)
            if resolved.exists():
                return resolved

        safe_name = sanitize_filename(filename or "plant-upload.jpg")
        target = output_dir / f"input_{safe_name}"
        target.write_bytes(image_bytes)
        return target

    def _annotate_analysis_paths(self, analysis: dict[str, Any], *, input_path: Path, output_dir: Path) -> dict[str, Any]:
        # Convert local artifact paths into stable API-facing paths and URLs so
        # the frontend can render the intermediate CV outputs directly.
        analysis_run_path = output_dir.relative_to(self.uploads_root).as_posix()
        analysis["analysis_run_path"] = analysis_run_path
        analysis["analysis_json_path"] = f"{analysis_run_path}/analysis.json"
        analysis["analysis_json_url"] = f"{settings.backend_public_url.rstrip('/')}/uploads/{analysis['analysis_json_path']}"

        if input_path.is_relative_to(self.uploads_root):
            input_relative_path = input_path.relative_to(self.uploads_root).as_posix()
            analysis["input"]["stored_input_path"] = input_relative_path
            analysis["input"]["stored_input_url"] = (
                f"{settings.backend_public_url.rstrip('/')}/uploads/{input_relative_path}"
            )

        for output in analysis.get("outputs", []):
            relative_path = f"{analysis_run_path}/{output['relative_path']}"
            output["image_path"] = relative_path
            output["image_url"] = f"{settings.backend_public_url.rstrip('/')}/uploads/{relative_path}"

        roi_info = analysis.get("roi")
        if roi_info:
            for key in ("scene_context_relative_path", "context_crop_relative_path", "classifier_primary_relative_path"):
                relative = roi_info.get(key)
                if not relative:
                    continue
                relative_path = f"{analysis_run_path}/{relative}"
                roi_info[key.replace("_relative_path", "_path")] = relative_path
                roi_info[key.replace("_relative_path", "_url")] = (
                    f"{settings.backend_public_url.rstrip('/')}/uploads/{relative_path}"
                )

        return analysis

    @staticmethod
    def _report_progress(
        progress_callback: ProgressCallback | None,
        stage_key: str,
        progress_percent: float,
        stage_label: str,
    ) -> None:
        if progress_callback:
            progress_callback(stage_key, progress_percent, stage_label)


@lru_cache
def get_inference_service() -> BaseInferenceService:
    if settings.enable_mock_inference:
        return MockInferenceService()
    return PlantVisionInferenceService(fallback_service=MockInferenceService())

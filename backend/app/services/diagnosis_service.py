"""Diagnosis orchestration service."""

from fastapi import UploadFile

from app.core.exceptions import AppException
from app.core.logging import get_logger
from app.core.security import AuthenticatedUser
from app.schemas.diagnosis import DiagnosisRead

logger = get_logger(__name__)


class DiagnosisService:
    def __init__(self, db_client, storage_service, inference_service, advisory_service) -> None:
        self.db_client = db_client
        self.storage_service = storage_service
        self.inference_service = inference_service
        self.advisory_service = advisory_service

    async def create_from_upload(
        self,
        *,
        upload: UploadFile,
        user: AuthenticatedUser,
        plant_id: str | None,
        farm_notes: str | None,
        crop_hint: str | None = None,
    ) -> DiagnosisRead:
        image_bytes, image_path, image_url = await self.storage_service.upload_diagnosis_image(upload, user.user_id)
        inference = await self.inference_service.predict(
            image_bytes=image_bytes,
            filename=upload.filename or "upload.bin",
            content_type=upload.content_type or "application/octet-stream",
            stored_image_path=image_path,
            crop_hint=crop_hint,
        )
        advisory = self.advisory_service.generate(
            crop=inference.predicted_crop,
            disease=inference.predicted_disease,
            health_status=inference.health_status,
            confidence=inference.confidence_score,
            severity=inference.severity_level,
            urgency=inference.urgency_level,
            farm_notes=farm_notes,
            analysis_payload=inference.raw_prediction_json,
        )

        payload = {
            "user_id": user.user_id,
            "plant_id": plant_id,
            "image_url": image_url,
            "image_path": image_path,
            "predicted_crop": inference.predicted_crop,
            "health_status": inference.health_status.value,
            "predicted_disease": inference.predicted_disease,
            "confidence_score": inference.confidence_score,
            "severity_level": inference.severity_level.value,
            "urgency_level": inference.urgency_level.value,
            "model_version": inference.model_version,
            "raw_prediction_json": inference.raw_prediction_json,
            "ai_advice_text": advisory.advisory_text,
            "advisory_payload": advisory.model_dump(),
        }

        try:
            result = self.db_client.table("diagnoses").insert(payload).execute()
            rows = result.data or []
            if not rows:
                raise AppException(message="Diagnosis insert returned no rows.", code="diagnosis_insert_failed")

            diagnosis = DiagnosisRead.model_validate(rows[0])
            self._create_notification_if_needed(user.user_id, plant_id, diagnosis)
            return diagnosis
        except Exception:
            self.storage_service.delete_file(image_path)
            raise

    def _create_notification_if_needed(self, user_id: str, plant_id: str | None, diagnosis: DiagnosisRead) -> None:
        if diagnosis.health_status.value == "healthy":
            return

        try:
            self.db_client.table("notifications").insert(
                {
                    "user_id": user_id,
                    "type": "diagnosis",
                    "title": "New plant health alert",
                    "message": f"{diagnosis.predicted_crop.title()} scan indicates {diagnosis.predicted_disease or diagnosis.health_status.value}.",
                    "related_diagnosis_id": diagnosis.id,
                    "related_plant_id": plant_id,
                }
            ).execute()
        except Exception as exc:
            logger.warning("notification_create_failed", extra={"reason": str(exc), "diagnosis_id": diagnosis.id})

"""Diagnosis schemas."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import HealthStatus, SeverityLevel, UrgencyLevel


class AdvisoryPayload(BaseModel):
    advisory_text: str
    advisory_source: str = "rules"
    llm_model: str | None = None
    what_happened: str
    likely_cause: str
    severity_summary: str
    why_it_matters: str
    treatment_steps: list[str]
    action_explanations: list[str] = Field(default_factory=list)
    prevention_tips: list[str]
    urgency_guidance: str
    follow_up_recommendation: str


class InferencePayload(BaseModel):
    predicted_crop: str
    health_status: HealthStatus
    predicted_disease: str | None = None
    confidence_score: float = Field(ge=0, le=1)
    severity_level: SeverityLevel
    urgency_level: UrgencyLevel
    model_version: str
    raw_prediction_json: dict


class InferenceJobStatusPayload(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    stage_key: str
    stage_label: str
    progress_percent: float = Field(ge=0, le=100)
    elapsed_seconds: float = Field(ge=0)
    estimated_total_seconds: float | None = Field(default=None, ge=0)
    remaining_seconds: float | None = Field(default=None, ge=0)
    result: InferencePayload | None = None
    advisory: AdvisoryPayload | None = None
    error_message: str | None = None
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None


class DiagnosisCreateManual(BaseModel):
    plant_id: str | None = None
    image_url: str = Field(max_length=500)
    image_path: str = Field(max_length=500)
    predicted_crop: str = Field(min_length=2, max_length=80)
    health_status: HealthStatus
    predicted_disease: str | None = Field(default=None, max_length=120)
    confidence_score: float = Field(ge=0, le=1)
    severity_level: SeverityLevel
    urgency_level: UrgencyLevel
    model_version: str = Field(min_length=2, max_length=80)
    raw_prediction_json: dict = Field(default_factory=dict)
    ai_advice_text: str = Field(min_length=5, max_length=5000)
    advisory_payload: dict = Field(default_factory=dict)


class DiagnosisRead(BaseModel):
    id: str
    user_id: str
    plant_id: str | None = None
    image_url: str
    image_path: str
    predicted_crop: str
    health_status: HealthStatus
    predicted_disease: str | None = None
    confidence_score: float
    severity_level: SeverityLevel
    urgency_level: UrgencyLevel
    model_version: str
    raw_prediction_json: dict
    ai_advice_text: str
    advisory_payload: dict
    created_at: str


class DiagnosisListItem(DiagnosisRead):
    pass


class DiagnosisDeleteResponse(BaseModel):
    id: str
    deleted: bool


class UploadDiagnosisResponse(BaseModel):
    diagnosis: DiagnosisRead

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "diagnosis": {
                    "id": "uuid",
                    "user_id": "uuid",
                    "plant_id": "uuid",
                    "image_url": "http://localhost:8000/uploads/diagnosis-images/user-id/file.png",
                    "image_path": "user-id/2026/04/file.png",
                    "predicted_crop": "tomato",
                    "health_status": "diseased",
                    "predicted_disease": "early_blight",
                    "confidence_score": 0.92,
                    "severity_level": "high",
                    "urgency_level": "urgent",
                    "model_version": "mock-cropcare-v1",
                    "raw_prediction_json": {"top_prediction": "early_blight"},
                    "ai_advice_text": "Apply fungicide and isolate affected leaves.",
                    "advisory_payload": {
                        "treatment_steps": ["Remove infected leaves", "Apply copper fungicide"],
                        "prevention_tips": ["Avoid overhead watering"]
                    },
                    "created_at": "2026-04-16T10:00:00+00:00"
                }
            }
        }
    )

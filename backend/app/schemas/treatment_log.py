"""Treatment log schemas."""

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import TreatmentStatus


class TreatmentLogCreate(BaseModel):
    plant_id: str
    diagnosis_id: str | None = None
    treatment_action: str = Field(min_length=3, max_length=200)
    notes: str | None = Field(default=None, max_length=2000)
    applied_at: str
    follow_up_date: str | None = None
    status: TreatmentStatus = TreatmentStatus.planned

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "plant_id": "plant-uuid",
                "diagnosis_id": "diagnosis-uuid",
                "treatment_action": "Applied copper fungicide",
                "notes": "Sprayed in the evening to reduce leaf burn.",
                "applied_at": "2026-04-16T10:30:00Z",
                "follow_up_date": "2026-04-20",
                "status": "applied",
            }
        }
    )


class TreatmentLogUpdate(BaseModel):
    plant_id: str | None = None
    diagnosis_id: str | None = None
    treatment_action: str | None = Field(default=None, min_length=3, max_length=200)
    notes: str | None = Field(default=None, max_length=2000)
    applied_at: str | None = None
    follow_up_date: str | None = None
    status: TreatmentStatus | None = None


class TreatmentLogRead(BaseModel):
    id: str
    user_id: str
    plant_id: str
    diagnosis_id: str | None = None
    treatment_action: str
    notes: str | None = None
    applied_at: str
    follow_up_date: str | None = None
    status: TreatmentStatus
    created_at: str

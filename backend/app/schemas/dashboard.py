"""Dashboard schemas."""

from pydantic import BaseModel

from app.schemas.diagnosis import DiagnosisListItem


class HealthDistributionItem(BaseModel):
    health_status: str
    count: int


class DashboardSummary(BaseModel):
    total_plants: int
    total_diagnoses: int
    recent_diagnoses: list[DiagnosisListItem]
    active_alerts: int
    treatment_follow_ups_due: int
    crop_health_distribution: list[HealthDistributionItem]

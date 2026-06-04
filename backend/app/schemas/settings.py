"""Persisted user settings schemas."""

from pydantic import BaseModel


class SettingsBase(BaseModel):
    disease_detection_alerts: bool = True
    treatment_reminders: bool = True
    weekly_health_reports: bool = True
    ai_tips_recommendations: bool = False
    auto_generate_treatment_plans: bool = True
    seasonal_recommendations: bool = True
    detailed_analysis_mode: bool = False
    share_anonymized_data: bool = False
    keep_detection_history: bool = True
    auto_detect_crop_type: bool = True
    auto_save_scans: bool = True


class SettingsRead(SettingsBase):
    user_id: str
    created_at: str
    updated_at: str


class SettingsUpdate(BaseModel):
    disease_detection_alerts: bool | None = None
    treatment_reminders: bool | None = None
    weekly_health_reports: bool | None = None
    ai_tips_recommendations: bool | None = None
    auto_generate_treatment_plans: bool | None = None
    seasonal_recommendations: bool | None = None
    detailed_analysis_mode: bool | None = None
    share_anonymized_data: bool | None = None
    keep_detection_history: bool | None = None
    auto_detect_crop_type: bool | None = None
    auto_save_scans: bool | None = None

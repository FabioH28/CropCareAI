"""User settings routes."""

from fastapi import APIRouter, Depends

from app.api.deps import get_db_client
from app.core.security import AuthenticatedUser, get_current_user
from app.schemas.common import ApiResponse
from app.schemas.settings import SettingsRead, SettingsUpdate

router = APIRouter()


def _default_settings_payload(user_id: str) -> dict[str, object]:
    return {
        "user_id": user_id,
        "disease_detection_alerts": True,
        "treatment_reminders": True,
        "weekly_health_reports": True,
        "ai_tips_recommendations": False,
        "auto_generate_treatment_plans": True,
        "seasonal_recommendations": True,
        "detailed_analysis_mode": False,
        "share_anonymized_data": False,
        "keep_detection_history": True,
        "auto_detect_crop_type": True,
        "auto_save_scans": True,
    }


def _fetch_or_create_settings(db, user_id: str) -> dict[str, object]:
    rows = db.table("user_preferences").select("*").eq("user_id", user_id).limit(1).execute().data or []
    if rows:
        return rows[0]

    created = db.table("user_preferences").insert(_default_settings_payload(user_id)).execute().data or []
    return created[0]


@router.get("", response_model=ApiResponse[SettingsRead])
def fetch_settings(
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db_client),
) -> ApiResponse[SettingsRead]:
    record = _fetch_or_create_settings(db, current_user.user_id)
    return ApiResponse.success_response(
        data=SettingsRead.model_validate(record),
        message="Settings fetched successfully.",
    )


@router.patch("", response_model=ApiResponse[SettingsRead])
def update_settings(
    payload: SettingsUpdate,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db_client),
) -> ApiResponse[SettingsRead]:
    _fetch_or_create_settings(db, current_user.user_id)
    updates = payload.model_dump(exclude_none=True)
    rows = db.table("user_preferences").update(updates).eq("user_id", current_user.user_id).execute().data or []
    return ApiResponse.success_response(
        data=SettingsRead.model_validate(rows[0]),
        message="Settings updated successfully.",
    )

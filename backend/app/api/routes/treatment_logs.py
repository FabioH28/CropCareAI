"""Treatment log routes."""

from fastapi import APIRouter, Depends, Query, status

from app.api.deps import get_db_client
from app.core.exceptions import NotFoundException
from app.core.security import AuthenticatedUser, get_current_user
from app.db.database import fetch_owned_record
from app.schemas.common import ApiResponse, ListResponse
from app.schemas.treatment_log import TreatmentLogCreate, TreatmentLogRead, TreatmentLogUpdate

router = APIRouter()


def _create_treatment_notification(db, *, user_id: str, plant_name: str, action: str, follow_up_date: str | None) -> None:
    follow_up_suffix = f" Follow-up date: {follow_up_date}." if follow_up_date else ""
    db.table("notifications").insert(
        {
            "user_id": user_id,
            "type": "treatment",
            "title": f"Treatment logged for {plant_name}",
            "message": f"{action} was saved for {plant_name}.{follow_up_suffix}",
        }
    ).execute()


@router.post("", response_model=ApiResponse[TreatmentLogRead], status_code=status.HTTP_201_CREATED)
def create_treatment_log(
    payload: TreatmentLogCreate,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db_client),
) -> ApiResponse[TreatmentLogRead]:
    plant_record = fetch_owned_record(db, "plants", payload.plant_id, current_user.user_id)
    if payload.diagnosis_id:
        fetch_owned_record(db, "diagnoses", payload.diagnosis_id, current_user.user_id)

    record = payload.model_dump(exclude_none=True)
    record["user_id"] = current_user.user_id
    result = db.table("treatment_logs").insert(record).execute()
    rows = result.data or []
    if not rows:
        raise NotFoundException(message="Treatment log could not be created.", code="treatment_log_create_failed")

    _create_treatment_notification(
        db,
        user_id=current_user.user_id,
        plant_name=str(plant_record.get("custom_name") or plant_record.get("crop_type") or "plant"),
        action=payload.treatment_action,
        follow_up_date=payload.follow_up_date,
    )

    return ApiResponse.success_response(
        data=TreatmentLogRead.model_validate(rows[0]),
        message="Treatment log created successfully.",
    )


@router.get("", response_model=ApiResponse[ListResponse[TreatmentLogRead]])
def list_treatment_logs(
    plant_id: str | None = Query(default=None),
    status_filter: str | None = Query(default=None, alias="status"),
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db_client),
) -> ApiResponse[ListResponse[TreatmentLogRead]]:
    query = db.table("treatment_logs").select("*").eq("user_id", current_user.user_id).order("applied_at", desc=True)
    if plant_id:
        query = query.eq("plant_id", plant_id)
    if status_filter:
        query = query.eq("status", status_filter)

    result = query.execute()
    items = [TreatmentLogRead.model_validate(row) for row in (result.data or [])]
    return ApiResponse.success_response(
        data=ListResponse(items=items, total=len(items)),
        message="Treatment logs fetched successfully.",
    )


@router.get("/{treatment_log_id}", response_model=ApiResponse[TreatmentLogRead])
def get_treatment_log(
    treatment_log_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db_client),
) -> ApiResponse[TreatmentLogRead]:
    record = fetch_owned_record(db, "treatment_logs", treatment_log_id, current_user.user_id)
    return ApiResponse.success_response(
        data=TreatmentLogRead.model_validate(record),
        message="Treatment log fetched successfully.",
    )


@router.patch("/{treatment_log_id}", response_model=ApiResponse[TreatmentLogRead])
def update_treatment_log(
    treatment_log_id: str,
    payload: TreatmentLogUpdate,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db_client),
) -> ApiResponse[TreatmentLogRead]:
    fetch_owned_record(db, "treatment_logs", treatment_log_id, current_user.user_id)

    updates = payload.model_dump(exclude_none=True)
    if "plant_id" in updates:
        fetch_owned_record(db, "plants", updates["plant_id"], current_user.user_id)
    if "diagnosis_id" in updates and updates["diagnosis_id"]:
        fetch_owned_record(db, "diagnoses", updates["diagnosis_id"], current_user.user_id)

    result = (
        db.table("treatment_logs")
        .update(updates)
        .eq("id", treatment_log_id)
        .eq("user_id", current_user.user_id)
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise NotFoundException(message="Treatment log not found.", code="treatment_log_not_found")

    return ApiResponse.success_response(
        data=TreatmentLogRead.model_validate(rows[0]),
        message="Treatment log updated successfully.",
    )


@router.delete("/{treatment_log_id}", response_model=ApiResponse[dict])
def delete_treatment_log(
    treatment_log_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db_client),
) -> ApiResponse[dict]:
    fetch_owned_record(db, "treatment_logs", treatment_log_id, current_user.user_id)
    db.table("treatment_logs").delete().eq("id", treatment_log_id).eq("user_id", current_user.user_id).execute()
    return ApiResponse.success_response(
        data={"id": treatment_log_id, "deleted": True},
        message="Treatment log deleted successfully.",
    )

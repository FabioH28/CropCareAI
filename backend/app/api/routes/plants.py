"""Plant management routes."""

from fastapi import APIRouter, Depends, Query, status

from app.api.deps import get_db_client
from app.core.exceptions import NotFoundException, ValidationException
from app.core.security import AuthenticatedUser, get_current_user
from app.db.database import fetch_owned_record
from app.schemas.common import ApiResponse, ListResponse
from app.schemas.plant import PlantCreate, PlantListItem, PlantRead, PlantUpdate

router = APIRouter()


@router.post("", response_model=ApiResponse[PlantRead], status_code=status.HTTP_201_CREATED)
def create_plant(
    payload: PlantCreate,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db_client),
) -> ApiResponse[PlantRead]:
    record = payload.model_dump(exclude_none=True)
    record["user_id"] = current_user.user_id

    if payload.farm_id:
        fetch_owned_record(db, "farms", payload.farm_id, current_user.user_id)

    result = db.table("plants").insert(record).execute()
    rows = result.data or []
    if not rows:
        raise ValidationException(message="Plant could not be created.", code="plant_create_failed")

    return ApiResponse.success_response(
        data=PlantRead.model_validate(rows[0]),
        message="Plant created successfully.",
    )


@router.get("", response_model=ApiResponse[ListResponse[PlantListItem]])
def list_plants(
    crop_type: str | None = Query(default=None),
    status_filter: str | None = Query(default=None, alias="status"),
    include_archived: bool = Query(default=False),
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db_client),
) -> ApiResponse[ListResponse[PlantListItem]]:
    query = db.table("plants").select("*").eq("user_id", current_user.user_id).order("created_at", desc=True)
    if crop_type:
        query = query.eq("crop_type", crop_type)
    if status_filter:
        query = query.eq("status", status_filter)
    if not include_archived:
        query = query.neq("status", "archived")

    result = query.execute()
    items = [PlantListItem.model_validate(row) for row in (result.data or [])]
    return ApiResponse.success_response(
        data=ListResponse(items=items, total=len(items)),
        message="Plants fetched successfully.",
    )


@router.get("/{plant_id}", response_model=ApiResponse[PlantRead])
def get_plant(
    plant_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db_client),
) -> ApiResponse[PlantRead]:
    record = fetch_owned_record(db, "plants", plant_id, current_user.user_id)
    return ApiResponse.success_response(data=PlantRead.model_validate(record), message="Plant fetched successfully.")


@router.patch("/{plant_id}", response_model=ApiResponse[PlantRead])
def update_plant(
    plant_id: str,
    payload: PlantUpdate,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db_client),
) -> ApiResponse[PlantRead]:
    fetch_owned_record(db, "plants", plant_id, current_user.user_id)

    updates = payload.model_dump(exclude_none=True)
    if "farm_id" in updates and updates["farm_id"]:
        fetch_owned_record(db, "farms", updates["farm_id"], current_user.user_id)

    result = db.table("plants").update(updates).eq("id", plant_id).eq("user_id", current_user.user_id).execute()
    rows = result.data or []
    if not rows:
        raise NotFoundException(message="Plant not found.", code="plant_not_found")

    return ApiResponse.success_response(
        data=PlantRead.model_validate(rows[0]),
        message="Plant updated successfully.",
    )


@router.delete("/{plant_id}", response_model=ApiResponse[PlantRead])
def archive_plant(
    plant_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db_client),
) -> ApiResponse[PlantRead]:
    fetch_owned_record(db, "plants", plant_id, current_user.user_id)
    result = (
        db.table("plants")
        .update({"status": "archived"})
        .eq("id", plant_id)
        .eq("user_id", current_user.user_id)
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise NotFoundException(message="Plant not found.", code="plant_not_found")

    return ApiResponse.success_response(
        data=PlantRead.model_validate(rows[0]),
        message="Plant archived successfully.",
    )

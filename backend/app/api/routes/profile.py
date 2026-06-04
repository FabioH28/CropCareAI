"""Profile management routes."""

from fastapi import APIRouter, Depends

from app.api.deps import get_db_client
from app.core.exceptions import NotFoundException
from app.core.security import AuthenticatedUser, get_current_user
from app.schemas.common import ApiResponse
from app.schemas.profile import ProfileRead, ProfileUpdate

router = APIRouter()


@router.get("", response_model=ApiResponse[ProfileRead])
def fetch_profile(
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db_client),
) -> ApiResponse[ProfileRead]:
    result = db.table("profiles").select("*").eq("id", current_user.user_id).limit(1).execute()
    rows = result.data or []
    if not rows:
        raise NotFoundException(message="Profile not found.", code="profile_not_found")

    return ApiResponse.success_response(
        data=ProfileRead.model_validate(rows[0]),
        message="Profile fetched successfully.",
    )


@router.patch("", response_model=ApiResponse[ProfileRead])
def update_profile(
    payload: ProfileUpdate,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db_client),
) -> ApiResponse[ProfileRead]:
    updates = payload.model_dump(exclude_none=True)
    result = db.table("profiles").update(updates).eq("id", current_user.user_id).execute()
    rows = result.data or []
    if not rows:
        raise NotFoundException(message="Profile not found.", code="profile_not_found")

    return ApiResponse.success_response(
        data=ProfileRead.model_validate(rows[0]),
        message="Profile updated successfully.",
    )

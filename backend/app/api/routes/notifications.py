"""Notification and alert routes."""

from fastapi import APIRouter, Depends, status

from app.api.deps import get_db_client
from app.core.exceptions import NotFoundException
from app.core.security import AuthenticatedUser, get_current_user
from app.db.database import fetch_owned_record
from app.schemas.common import ApiResponse, ListResponse
from app.schemas.notification import NotificationRead, NotificationStatusUpdate

router = APIRouter()


@router.get("", response_model=ApiResponse[ListResponse[NotificationRead]])
def list_notifications(
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db_client),
) -> ApiResponse[ListResponse[NotificationRead]]:
    result = (
        db.table("notifications")
        .select("*")
        .eq("user_id", current_user.user_id)
        .order("created_at", desc=True)
        .execute()
    )
    items = [NotificationRead.model_validate(row) for row in (result.data or [])]
    return ApiResponse.success_response(
        data=ListResponse(items=items, total=len(items)),
        message="Notifications fetched successfully.",
    )


@router.post("/{notification_id}/read", response_model=ApiResponse[NotificationRead], status_code=status.HTTP_200_OK)
def mark_notification_as_read(
    notification_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db_client),
) -> ApiResponse[NotificationRead]:
    fetch_owned_record(db, "notifications", notification_id, current_user.user_id)
    result = (
        db.table("notifications")
        .update(NotificationStatusUpdate(read=True).model_dump())
        .eq("id", notification_id)
        .eq("user_id", current_user.user_id)
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise NotFoundException(message="Notification not found.", code="notification_not_found")

    return ApiResponse.success_response(
        data=NotificationRead.model_validate(rows[0]),
        message="Notification marked as read.",
    )


@router.post("/read-all", response_model=ApiResponse[dict])
def mark_all_notifications_as_read(
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db_client),
) -> ApiResponse[dict]:
    result = (
        db.table("notifications")
        .update(NotificationStatusUpdate(read=True).model_dump())
        .eq("user_id", current_user.user_id)
        .eq("read", False)
        .execute()
    )
    updated = len(result.data or [])
    return ApiResponse.success_response(
        data={"updated_count": updated},
        message="All notifications marked as read.",
    )

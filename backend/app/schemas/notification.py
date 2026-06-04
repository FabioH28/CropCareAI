"""Notification schemas."""

from pydantic import BaseModel

from app.models.enums import NotificationType


class NotificationRead(BaseModel):
    id: str
    user_id: str
    type: NotificationType
    title: str
    message: str
    read: bool
    related_diagnosis_id: str | None = None
    related_plant_id: str | None = None
    created_at: str


class NotificationStatusUpdate(BaseModel):
    read: bool

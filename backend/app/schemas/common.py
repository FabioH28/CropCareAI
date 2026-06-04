"""Common response and utility schemas."""

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    success: bool = True
    message: str
    data: T | None = None
    meta: dict[str, Any] | None = None

    @classmethod
    def success_response(cls, data: T | None, message: str, meta: dict[str, Any] | None = None) -> "ApiResponse[T]":
        return cls(success=True, message=message, data=data, meta=meta)


class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    error: dict[str, Any]


class ListResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int = Field(ge=0)


class HealthCheckData(BaseModel):
    status: str
    service: str
    environment: str

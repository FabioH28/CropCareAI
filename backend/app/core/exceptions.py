"""Exception types and handlers for consistent API errors."""

from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core.logging import get_logger
from app.schemas.common import ErrorResponse

logger = get_logger(__name__)


class AppException(Exception):
    def __init__(
        self,
        message: str,
        code: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = dict(details or {})
        super().__init__(message)


class UnauthorizedException(AppException):
    def __init__(self, message: str = "Unauthorized.", code: str = "unauthorized", details=None) -> None:
        super().__init__(message=message, code=code, status_code=status.HTTP_401_UNAUTHORIZED, details=details)


class NotFoundException(AppException):
    def __init__(self, message: str = "Resource not found.", code: str = "not_found", details=None) -> None:
        super().__init__(message=message, code=code, status_code=status.HTTP_404_NOT_FOUND, details=details)


class ValidationException(AppException):
    def __init__(self, message: str = "Validation error.", code: str = "validation_error", details=None) -> None:
        super().__init__(message=message, code=code, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, details=details)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppException)
    async def handle_app_exception(_: Request, exc: AppException) -> JSONResponse:
        logger.warning("app_exception", extra={"code": exc.code, "details": exc.details})
        payload = ErrorResponse(
            success=False,
            message=exc.message,
            error={"code": exc.code, "details": exc.details or None},
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump(exclude_none=True))

    @app.exception_handler(Exception)
    async def handle_unexpected_exception(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_exception")
        payload = ErrorResponse(
            success=False,
            message="An unexpected error occurred.",
            error={"code": "internal_server_error", "details": {"reason": str(exc)}},
        )
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=payload.model_dump())

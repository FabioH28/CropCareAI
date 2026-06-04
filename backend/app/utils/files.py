"""File validation helpers for uploads."""

from pathlib import Path

from app.core.config import get_settings
from app.core.exceptions import ValidationException

settings = get_settings()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def sanitize_filename(filename: str) -> str:
    cleaned = Path(filename).name.replace(" ", "_")
    return "".join(char for char in cleaned if char.isalnum() or char in {"_", "-", "."})


def validate_upload(*, filename: str, content_type: str, file_size: int) -> None:
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValidationException(
            message="Unsupported file extension.",
            code="invalid_file_extension",
            details={"allowed_extensions": sorted(ALLOWED_EXTENSIONS)},
        )

    if content_type.lower() not in settings.allowed_image_types:
        raise ValidationException(
            message="Unsupported content type.",
            code="invalid_content_type",
            details={"allowed_types": settings.allowed_image_types},
        )

    if file_size <= 0:
        raise ValidationException(message="Uploaded file is empty.", code="empty_file")

    if file_size > settings.max_upload_size_bytes:
        raise ValidationException(
            message="Uploaded file exceeds the maximum allowed size.",
            code="file_too_large",
            details={"max_upload_size_bytes": settings.max_upload_size_bytes},
        )

"""Local filesystem storage helpers."""

from functools import lru_cache
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import get_settings
from app.core.exceptions import AppException
from app.core.logging import get_logger
from app.utils.files import sanitize_filename, validate_upload

logger = get_logger(__name__)
settings = get_settings()


class StorageService:
    def __init__(self) -> None:
        self.root = Path(settings.uploads_root).resolve()
        self.bucket = settings.upload_bucket_name
        self.root.mkdir(parents=True, exist_ok=True)

    async def upload_diagnosis_image(self, upload: UploadFile, user_id: str) -> tuple[bytes, str, str]:
        image_bytes = await upload.read()
        validate_upload(
            filename=upload.filename or "",
            content_type=upload.content_type or "",
            file_size=len(image_bytes),
        )

        safe_name = sanitize_filename(upload.filename or "upload.bin")
        image_path = f"{self.bucket}/{user_id}/{uuid4().hex[:8]}/{uuid4().hex}_{safe_name}"
        logger.info("uploading_image_to_storage", extra={"path": image_path, "user_id": user_id})

        try:
            destination = self.root / Path(image_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(image_bytes)
        except Exception as exc:
            raise AppException(
                message="Unable to upload image to storage.",
                code="storage_upload_failed",
                details={"reason": str(exc)},
            ) from exc

        normalized_path = image_path.replace("\\", "/")
        public_url = f"{settings.backend_public_url.rstrip('/')}/uploads/{normalized_path}"

        return image_bytes, image_path, public_url

    def delete_file(self, image_path: str) -> None:
        try:
            target = self.root / Path(image_path)
            if target.is_file():
                target.unlink()
            elif target.is_dir():
                rmtree(target, ignore_errors=True)
        except Exception as exc:
            logger.warning("storage_delete_failed", extra={"path": image_path, "reason": str(exc)})


@lru_cache
def get_storage_service() -> StorageService:
    return StorageService()

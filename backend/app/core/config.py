"""Application configuration."""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="CropCare AI API", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_debug: bool = Field(default=True, alias="APP_DEBUG")
    api_v1_prefix: str = Field(default="/api/v1", alias="API_V1_PREFIX")
    backend_host: str = Field(default="0.0.0.0", alias="BACKEND_HOST")
    backend_port: int = Field(default=8000, alias="BACKEND_PORT")
    frontend_url: str = Field(default="http://localhost:5173", alias="FRONTEND_URL")
    cors_allow_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ],
        alias="CORS_ALLOW_ORIGINS",
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    database_url: str = Field(default="mysql+pymysql://root:cropcare_root@127.0.0.1:3308/cropcare_ai", alias="DATABASE_URL")
    backend_public_url: str = Field(default="http://localhost:8000", alias="BACKEND_PUBLIC_URL")
    uploads_root: str = Field(default="storage", alias="UPLOADS_ROOT")
    upload_bucket_name: str = Field(default="diagnosis-images", alias="UPLOAD_BUCKET_NAME")

    auth_secret_key: str = Field(default="change-me-local-dev-secret-please-replace", alias="AUTH_SECRET_KEY")
    auth_algorithm: str = Field(default="HS256", alias="AUTH_ALGORITHM")
    auth_access_token_expire_minutes: int = Field(default=60, alias="AUTH_ACCESS_TOKEN_EXPIRE_MINUTES")
    auth_refresh_token_expire_days: int = Field(default=14, alias="AUTH_REFRESH_TOKEN_EXPIRE_DAYS")

    max_upload_size_bytes: int = Field(default=5 * 1024 * 1024, alias="MAX_UPLOAD_SIZE_BYTES")
    allowed_image_types: list[str] = Field(default_factory=lambda: ["image/jpeg", "image/png", "image/webp"], alias="ALLOWED_IMAGE_TYPES")

    default_model_version: str = Field(default="plant-vision-cv-v2-leaf-first", alias="DEFAULT_MODEL_VERSION")
    enable_mock_inference: bool = Field(default=False, alias="ENABLE_MOCK_INFERENCE")
    enable_dl_classifier: bool = Field(default=True, alias="ENABLE_DL_CLASSIFIER")
    dl_checkpoint_path: str | None = Field(default=None, alias="DL_CHECKPOINT_PATH")
    dl_confidence_threshold: float = Field(default=0.65, alias="DL_CONFIDENCE_THRESHOLD")
    dl_tta_passes: int = Field(default=4, alias="DL_TTA_PASSES")
    dl_apple_specialist_path: str | None = Field(default="ml/plant/artifacts/checkpoints/apple_specialist.pt", alias="DL_APPLE_SPECIALIST_PATH")
    dl_tomato_specialist_path: str | None = Field(default="ml/plant/artifacts/checkpoints/tomato_specialist.pt", alias="DL_TOMATO_SPECIALIST_PATH")
    dl_grape_specialist_path: str | None = Field(default="ml/plant/artifacts/checkpoints/grape_specialist.pt", alias="DL_GRAPE_SPECIALIST_PATH")
    dl_corn_specialist_path: str | None = Field(default="ml/plant/artifacts/checkpoints/corn_specialist.pt", alias="DL_CORN_SPECIALIST_PATH")
    dl_pepper_specialist_path: str | None = Field(default="ml/plant/artifacts/checkpoints/pepper_specialist.pt", alias="DL_PEPPER_SPECIALIST_PATH")
    dl_potato_specialist_path: str | None = Field(default="ml/plant/artifacts/checkpoints/potato_specialist.pt", alias="DL_POTATO_SPECIALIST_PATH")
    dl_cascade_routing_threshold: float = Field(default=0.50, alias="DL_CASCADE_ROUTING_THRESHOLD")
    enable_rules_advisory: bool = Field(default=True, alias="ENABLE_RULES_ADVISORY")
    enable_llm_advisory: bool = Field(default=True, alias="ENABLE_LLM_ADVISORY")
    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")
    llm_base_url: str = Field(default="http://localhost:11434", alias="LLM_BASE_URL")
    llm_model: str = Field(default="llama3:8b", alias="LLM_MODEL")
    llm_timeout_seconds: float = Field(default=90.0, alias="LLM_TIMEOUT_SECONDS")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def split_cors_origins(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("allowed_image_types", mode="before")
    @classmethod
    def split_allowed_types(cls, value):
        if isinstance(value, str):
            return [item.strip().lower() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()

"""FastAPI entrypoint for the CropCare AI backend."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.schemas.common import ApiResponse, HealthCheckData

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("starting_cropcare_api env=%s", settings.app_env)
    yield
    logger.info("stopping_cropcare_api env=%s", settings.app_env)


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Production-oriented MVP backend for CropCare AI.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)
app.include_router(api_router, prefix=settings.api_v1_prefix)

uploads_root = Path(settings.uploads_root).resolve()
uploads_root.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=uploads_root), name="uploads")


@app.get("/", response_model=ApiResponse[HealthCheckData], tags=["system"])
async def root() -> ApiResponse[HealthCheckData]:
    return ApiResponse.success_response(
        data=HealthCheckData(status="ok", service=settings.app_name, environment=settings.app_env),
        message="CropCare AI API is running.",
    )


@app.get("/healthz", response_model=ApiResponse[HealthCheckData], tags=["system"])
async def healthcheck() -> ApiResponse[HealthCheckData]:
    return ApiResponse.success_response(
        data=HealthCheckData(status="ok", service=settings.app_name, environment=settings.app_env),
        message="Healthcheck passed.",
    )

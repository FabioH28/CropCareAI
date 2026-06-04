"""Top-level API router composition."""

from fastapi import APIRouter

from app.api.routes import advisor, auth, dashboard, diagnoses, notifications, plants, profile, settings, treatment_logs

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(profile.router, prefix="/profile", tags=["profile"])
api_router.include_router(settings.router, prefix="/settings", tags=["settings"])
api_router.include_router(plants.router, prefix="/plants", tags=["plants"])
api_router.include_router(diagnoses.router, prefix="/diagnoses", tags=["diagnoses"])
api_router.include_router(treatment_logs.router, prefix="/treatment-logs", tags=["treatment_logs"])
api_router.include_router(notifications.router, prefix="/notifications", tags=["notifications"])
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
api_router.include_router(advisor.router, prefix="/advisor", tags=["advisor"])

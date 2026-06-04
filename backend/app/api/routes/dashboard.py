"""Dashboard summary routes."""

from fastapi import APIRouter, Depends

from app.api.deps import get_db_client
from app.core.security import AuthenticatedUser, get_current_user
from app.schemas.common import ApiResponse
from app.schemas.dashboard import DashboardSummary
from app.services.dashboard_service import DashboardService

router = APIRouter()


@router.get("/summary", response_model=ApiResponse[DashboardSummary])
def get_dashboard_summary(
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db_client),
) -> ApiResponse[DashboardSummary]:
    service = DashboardService(db)
    summary = service.build_summary(current_user.user_id)
    return ApiResponse.success_response(data=summary, message="Dashboard summary fetched successfully.")

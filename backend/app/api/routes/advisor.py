"""Plant advisor chat route."""

from fastapi import APIRouter, Depends

from app.api.deps import get_db_client
from app.core.security import AuthenticatedUser, get_current_user
from app.schemas.advisor import AdvisorChatRequest, AdvisorChatResponse
from app.schemas.common import ApiResponse
from app.services.advisor_chat_service import AdvisorChatService

router = APIRouter()


@router.post("/chat", response_model=ApiResponse[AdvisorChatResponse])
def advisor_chat(
    payload: AdvisorChatRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db_client),
) -> ApiResponse[AdvisorChatResponse]:
    service = AdvisorChatService(db)
    result = service.ask(
        user_id=current_user.user_id,
        message=payload.message,
        context_mode=payload.context_mode,
        model_mode=payload.model_mode,
        current_diagnosis=payload.current_diagnosis,
    )
    return ApiResponse.success_response(
        data=AdvisorChatResponse.model_validate(result),
        message="Advisor reply generated successfully.",
    )

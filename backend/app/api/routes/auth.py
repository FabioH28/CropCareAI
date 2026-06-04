"""Authentication routes backed by the project database and JWTs."""

from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, status

from app.api.deps import get_db_client
from app.core.auth import create_access_token, create_refresh_token, hash_password, verify_password
from app.core.config import get_settings
from app.core.exceptions import AppException, NotFoundException, UnauthorizedException
from app.core.security import AuthenticatedUser, get_current_user
from app.schemas.auth import (
    ChangePasswordRequest,
    ChangePasswordResponse,
    LoginRequest,
    LogoutResponse,
    PasswordResetRequest,
    PasswordResetRequestResponse,
    SessionTokens,
    SignUpRequest,
    UserSessionData,
)
from app.schemas.common import ApiResponse
from app.schemas.profile import ProfileRead

router = APIRouter()
settings = get_settings()


def _build_session_tokens(*, user_id: str, email: str) -> SessionTokens:
    access_token, expires_at = create_access_token(user_id=user_id, email=email)
    refresh_token, _ = create_refresh_token(user_id=user_id, email=email)
    return SessionTokens(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.auth_access_token_expire_minutes * 60,
        expires_at=expires_at,
    )


@router.post("/sign-up", response_model=ApiResponse[UserSessionData], status_code=status.HTTP_201_CREATED)
def sign_up(payload: SignUpRequest, db=Depends(get_db_client)) -> ApiResponse[UserSessionData]:
    email = payload.email.lower()
    existing = db.table("users").select("*").eq("email", email).limit(1).execute().data or []
    if existing:
        raise AppException(
            message="An account with this email already exists.",
            code="email_already_registered",
            status_code=status.HTTP_409_CONFLICT,
        )

    user_id = str(uuid4())
    confirmed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    with db.transaction():
        db.table("users").insert(
            {
                "id": user_id,
                "email": email,
                "password_hash": hash_password(payload.password.get_secret_value()),
                "email_confirmed_at": confirmed_at,
                "is_active": True,
            }
        ).execute()
        db.table("profiles").insert(
            {
                "id": user_id,
                "email": email,
                "full_name": payload.full_name,
                "farm_name": payload.farm_name,
                "phone": payload.phone,
            }
        ).execute()

    data = UserSessionData(
        user_id=user_id,
        email=email,
        email_confirmed_at=confirmed_at,
        session=_build_session_tokens(user_id=user_id, email=email),
    )
    return ApiResponse.success_response(data=data, message="Account created successfully.")


@router.post("/log-in", response_model=ApiResponse[UserSessionData])
def log_in(payload: LoginRequest, db=Depends(get_db_client)) -> ApiResponse[UserSessionData]:
    email = payload.email.lower()
    rows = db.table("users").select("*").eq("email", email).limit(1).execute().data or []
    if not rows:
        raise UnauthorizedException(
            message="Invalid email or password.",
            code="invalid_credentials",
        )

    user = rows[0]
    if not verify_password(payload.password.get_secret_value(), user.get("password_hash", "")):
        raise UnauthorizedException(
            message="Invalid email or password.",
            code="invalid_credentials",
        )

    data = UserSessionData(
        user_id=str(user["id"]),
        email=user.get("email") or email,
        email_confirmed_at=user.get("email_confirmed_at"),
        session=_build_session_tokens(user_id=str(user["id"]), email=user.get("email") or email),
    )
    return ApiResponse.success_response(data=data, message="Logged in successfully.")


@router.post("/log-out", response_model=ApiResponse[LogoutResponse])
def log_out(
    _: AuthenticatedUser = Depends(get_current_user),
    authorization: str | None = Header(default=None),
) -> ApiResponse[LogoutResponse]:
    if not authorization:
        raise UnauthorizedException(message="Authorization header is required.", code="missing_auth_header")

    return ApiResponse.success_response(
        data=LogoutResponse(logged_out=True, revoke_note="Discard the local JWT on the client after this response."),
        message="Logout acknowledged.",
    )


@router.post("/request-password-reset", response_model=ApiResponse[PasswordResetRequestResponse])
def request_password_reset(
    payload: PasswordResetRequest,
    db=Depends(get_db_client),
) -> ApiResponse[PasswordResetRequestResponse]:
    email = payload.email.lower()
    rows = db.table("users").select("*").eq("email", email).limit(1).execute().data or []
    if rows:
        db.table("password_reset_requests").insert({"user_id": rows[0]["id"], "email": email}).execute()

    return ApiResponse.success_response(
        data=PasswordResetRequestResponse(
            recorded=True,
            delivery_mode="manual-local",
            note="Password reset requests are recorded in the project database. This project does not send real emails.",
        ),
        message="If an account exists for that email, the reset request has been recorded.",
    )


@router.post("/change-password", response_model=ApiResponse[ChangePasswordResponse])
def change_password(
    payload: ChangePasswordRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db_client),
) -> ApiResponse[ChangePasswordResponse]:
    rows = db.table("users").select("*").eq("id", current_user.user_id).limit(1).execute().data or []
    if not rows:
        raise NotFoundException(message="User account not found.", code="user_not_found")

    user = rows[0]
    if not verify_password(payload.current_password.get_secret_value(), user.get("password_hash", "")):
        raise UnauthorizedException(message="Current password is incorrect.", code="invalid_current_password")

    db.table("users").update(
        {"password_hash": hash_password(payload.new_password.get_secret_value())}
    ).eq("id", current_user.user_id).execute()
    return ApiResponse.success_response(
        data=ChangePasswordResponse(updated=True),
        message="Password updated successfully.",
    )


@router.get("/me", response_model=ApiResponse[ProfileRead])
def get_me(
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db_client),
) -> ApiResponse[ProfileRead]:
    result = db.table("profiles").select("*").eq("id", current_user.user_id).limit(1).execute()
    rows = result.data or []
    if not rows:
        raise NotFoundException(message="Profile not found.", code="profile_not_found")

    profile = ProfileRead.model_validate(rows[0])
    return ApiResponse.success_response(data=profile, message="Authenticated user profile fetched.")

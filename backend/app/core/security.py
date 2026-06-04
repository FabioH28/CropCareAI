"""Security helpers and local JWT-backed auth dependencies."""

from dataclasses import dataclass

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.api.deps import get_db_client
from app.core.auth import decode_token
from app.core.exceptions import UnauthorizedException
from app.core.logging import get_logger

logger = get_logger(__name__)
http_bearer = HTTPBearer(auto_error=False)


@dataclass(slots=True)
class AuthenticatedUser:
    user_id: str
    email: str | None
    access_token: str


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(http_bearer),
    db=Depends(get_db_client),
) -> AuthenticatedUser:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise UnauthorizedException(message="Bearer token is required.", code="missing_bearer_token")

    token = credentials.credentials

    try:
        payload = decode_token(token, expected_type="access")
    except UnauthorizedException:
        raise
    except Exception as exc:
        logger.warning("local_auth_failed", extra={"reason": str(exc)})
        raise UnauthorizedException(message="Invalid or expired access token.", code="invalid_token") from exc

    rows = db.table("users").select("*").eq("id", payload.user_id).limit(1).execute().data or []
    if not rows:
        raise UnauthorizedException(message="Authenticated user could not be resolved.", code="user_resolution_failed")

    user = rows[0]
    return AuthenticatedUser(user_id=str(user["id"]), email=user.get("email"), access_token=token)

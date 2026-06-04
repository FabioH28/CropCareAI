"""Local authentication helpers for password hashing and JWT management."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt

from app.core.config import get_settings
from app.core.exceptions import UnauthorizedException

settings = get_settings()
PASSWORD_SCHEME = "pbkdf2_sha256"
PASSWORD_ITERATIONS = 390_000


@dataclass(slots=True)
class TokenPayload:
    user_id: str
    email: str | None
    token_type: str
    expires_at: int


def _utc_now() -> datetime:
    return datetime.now(UTC)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)
    encoded_salt = base64.urlsafe_b64encode(salt).decode("ascii")
    encoded_digest = base64.urlsafe_b64encode(digest).decode("ascii")
    return f"{PASSWORD_SCHEME}${PASSWORD_ITERATIONS}${encoded_salt}${encoded_digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        scheme, iterations_raw, encoded_salt, encoded_digest = stored_hash.split("$", 3)
        if scheme != PASSWORD_SCHEME:
            return False
        iterations = int(iterations_raw)
        salt = base64.urlsafe_b64decode(encoded_salt.encode("ascii"))
        expected = base64.urlsafe_b64decode(encoded_digest.encode("ascii"))
    except (ValueError, TypeError):
        return False

    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(candidate, expected)


def _build_token(*, user_id: str, email: str | None, token_type: str, expires_delta: timedelta) -> tuple[str, int]:
    expires_at = _utc_now() + expires_delta
    payload = {
        "sub": user_id,
        "email": email,
        "type": token_type,
        "iat": int(_utc_now().timestamp()),
        "exp": int(expires_at.timestamp()),
    }
    token = jwt.encode(payload, settings.auth_secret_key, algorithm=settings.auth_algorithm)
    return token, int(expires_at.timestamp())


def create_access_token(*, user_id: str, email: str | None) -> tuple[str, int]:
    return _build_token(
        user_id=user_id,
        email=email,
        token_type="access",
        expires_delta=timedelta(minutes=settings.auth_access_token_expire_minutes),
    )


def create_refresh_token(*, user_id: str, email: str | None) -> tuple[str, int]:
    return _build_token(
        user_id=user_id,
        email=email,
        token_type="refresh",
        expires_delta=timedelta(days=settings.auth_refresh_token_expire_days),
    )


def decode_token(token: str, *, expected_type: str = "access") -> TokenPayload:
    try:
        payload = jwt.decode(token, settings.auth_secret_key, algorithms=[settings.auth_algorithm])
    except jwt.PyJWTError as exc:
        raise UnauthorizedException(message="Invalid or expired access token.", code="invalid_token") from exc

    token_type = payload.get("type")
    user_id = payload.get("sub")
    expires_at = payload.get("exp")
    if token_type != expected_type or not user_id or not expires_at:
        raise UnauthorizedException(message="Invalid access token payload.", code="invalid_token_payload")

    return TokenPayload(
        user_id=str(user_id),
        email=payload.get("email"),
        token_type=str(token_type),
        expires_at=int(expires_at),
    )

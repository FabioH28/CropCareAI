"""Authentication schemas."""

from pydantic import BaseModel, ConfigDict, EmailStr, Field, SecretStr


class SignUpRequest(BaseModel):
    email: EmailStr
    password: SecretStr = Field(min_length=8)
    full_name: str = Field(min_length=2, max_length=120)
    farm_name: str | None = Field(default=None, max_length=120)
    phone: str | None = Field(default=None, max_length=40)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "farmer@example.com",
                "password": "StrongPass123!",
                "full_name": "Alex Farmer",
                "farm_name": "Green Valley Farm",
            }
        }
    )


class LoginRequest(BaseModel):
    email: EmailStr
    password: SecretStr = Field(min_length=8)

    model_config = ConfigDict(
        json_schema_extra={"example": {"email": "farmer@example.com", "password": "StrongPass123!"}}
    )


class SessionTokens(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int | None = None
    expires_at: int | None = None


class UserSessionData(BaseModel):
    user_id: str
    email: EmailStr
    email_confirmed_at: str | None = None
    session: SessionTokens | None = None


class LogoutResponse(BaseModel):
    logged_out: bool
    revoke_note: str


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetRequestResponse(BaseModel):
    recorded: bool
    delivery_mode: str
    note: str


class ChangePasswordRequest(BaseModel):
    current_password: SecretStr = Field(min_length=8)
    new_password: SecretStr = Field(min_length=8)


class ChangePasswordResponse(BaseModel):
    updated: bool

"""Profile schemas."""

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class ProfileRead(BaseModel):
    id: str
    full_name: str | None = None
    email: EmailStr
    farm_name: str | None = None
    phone: str | None = None
    location: str | None = None
    avatar_url: str | None = None
    created_at: str
    updated_at: str


class ProfileUpdate(BaseModel):
    full_name: str | None = Field(default=None, min_length=2, max_length=120)
    farm_name: str | None = Field(default=None, max_length=120)
    phone: str | None = Field(default=None, max_length=40)
    location: str | None = Field(default=None, max_length=200)
    avatar_url: str | None = Field(default=None, max_length=500)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "full_name": "Alex Farmer",
                "farm_name": "Green Valley Farm",
                "phone": "+49-555-0100",
                "location": "Brandenburg, Germany",
            }
        }
    )

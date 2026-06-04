"""Plant schemas."""

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import PlantStatus


class PlantBase(BaseModel):
    crop_type: str = Field(min_length=2, max_length=80)
    custom_name: str = Field(min_length=2, max_length=120)
    zone_or_field: str | None = Field(default=None, max_length=120)
    planted_at: str | None = None
    status: PlantStatus = PlantStatus.active
    notes: str | None = Field(default=None, max_length=2000)
    farm_id: str | None = None


class PlantCreate(PlantBase):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "crop_type": "tomato",
                "custom_name": "Tomato Row A",
                "zone_or_field": "Field 1",
                "status": "active",
                "notes": "South irrigation line nearby.",
            }
        }
    )


class PlantUpdate(BaseModel):
    crop_type: str | None = Field(default=None, min_length=2, max_length=80)
    custom_name: str | None = Field(default=None, min_length=2, max_length=120)
    zone_or_field: str | None = Field(default=None, max_length=120)
    planted_at: str | None = None
    status: PlantStatus | None = None
    notes: str | None = Field(default=None, max_length=2000)
    farm_id: str | None = None


class PlantRead(PlantBase):
    id: str
    user_id: str
    created_at: str
    updated_at: str


class PlantListItem(PlantRead):
    pass

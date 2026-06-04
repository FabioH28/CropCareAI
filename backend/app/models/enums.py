"""Shared domain enums."""

from enum import StrEnum


class PlantStatus(StrEnum):
    active = "active"
    monitoring = "monitoring"
    diseased = "diseased"
    archived = "archived"


class HealthStatus(StrEnum):
    healthy = "healthy"
    suspicious = "suspicious"
    diseased = "diseased"


class SeverityLevel(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class UrgencyLevel(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"
    urgent = "urgent"


class TreatmentStatus(StrEnum):
    planned = "planned"
    applied = "applied"
    monitoring = "monitoring"
    completed = "completed"


class NotificationType(StrEnum):
    diagnosis = "diagnosis"
    follow_up = "follow_up"
    treatment = "treatment"
    system = "system"

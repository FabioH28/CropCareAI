"""Dashboard aggregation service."""

from collections import Counter
from datetime import date

from app.schemas.dashboard import DashboardSummary, HealthDistributionItem
from app.schemas.diagnosis import DiagnosisListItem


class DashboardService:
    def __init__(self, db_client) -> None:
        self.db = db_client

    def build_summary(self, user_id: str) -> DashboardSummary:
        plants = self.db.table("plants").select("*").eq("user_id", user_id).neq("status", "archived").execute().data or []
        diagnoses = self.db.table("diagnoses").select("*").eq("user_id", user_id).order("created_at", desc=True).execute().data or []
        notifications = self.db.table("notifications").select("*").eq("user_id", user_id).eq("read", False).execute().data or []
        treatment_logs = self.db.table("treatment_logs").select("*").eq("user_id", user_id).execute().data or []

        due_count = 0
        today = date.today().isoformat()
        for log in treatment_logs:
            follow_up = log.get("follow_up_date")
            if follow_up and follow_up <= today and log.get("status") != "completed":
                due_count += 1

        distribution = Counter(item.get("health_status", "unknown") for item in diagnoses)
        health_items = [
            HealthDistributionItem(health_status=status, count=count)
            for status, count in sorted(distribution.items())
        ]

        recent = [DiagnosisListItem.model_validate(item) for item in diagnoses[:5]]
        return DashboardSummary(
            total_plants=len(plants),
            total_diagnoses=len(diagnoses),
            recent_diagnoses=recent,
            active_alerts=len(notifications),
            treatment_follow_ups_due=due_count,
            crop_health_distribution=health_items,
        )

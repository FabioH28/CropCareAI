"""Background preview jobs with stage-based progress snapshots."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4

from app.core.logging import get_logger
from app.schemas.diagnosis import AdvisoryPayload, InferenceJobStatusPayload, InferencePayload
from app.utils.files import validate_upload

logger = get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class _InferenceJobRecord:
    job_id: str
    status: str
    stage_key: str
    stage_label: str
    progress_percent: float
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: InferencePayload | None = None
    advisory: AdvisoryPayload | None = None
    error_message: str | None = None


class InferenceJobService:
    """Keeps lightweight in-memory status for plant preview jobs."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._jobs: dict[str, _InferenceJobRecord] = {}
        self._completed_durations: deque[float] = deque(maxlen=20)

    def create_job(self) -> InferenceJobStatusPayload:
        with self._lock:
            job_id = uuid4().hex
            record = _InferenceJobRecord(
                job_id=job_id,
                status="queued",
                stage_key="queued",
                stage_label="Waiting to start",
                progress_percent=0.0,
                created_at=_utcnow(),
            )
            self._jobs[job_id] = record
            self._trim_jobs_locked()
            return self._snapshot_locked(record)

    def get_job(self, job_id: str) -> InferenceJobStatusPayload | None:
        with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return None
            return self._snapshot_locked(record)

    def update_progress(
        self,
        job_id: str,
        *,
        stage_key: str,
        stage_label: str,
        progress_percent: float,
        status: str = "running",
    ) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if not record or record.status in {"completed", "failed"}:
                return

            if record.started_at is None and status == "running":
                record.started_at = _utcnow()

            record.status = status
            record.stage_key = stage_key
            record.stage_label = stage_label
            record.progress_percent = max(record.progress_percent, min(progress_percent, 100.0))

    def complete_job(
        self,
        job_id: str,
        *,
        result: InferencePayload,
        advisory: AdvisoryPayload,
    ) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return

            if record.started_at is None:
                record.started_at = _utcnow()

            record.status = "completed"
            record.stage_key = "completed"
            record.stage_label = "Analysis complete"
            record.progress_percent = 100.0
            record.completed_at = _utcnow()
            record.result = result
            record.advisory = advisory
            self._completed_durations.append(self._elapsed_seconds(record))

    def fail_job(self, job_id: str, *, message: str) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return

            if record.started_at is None:
                record.started_at = _utcnow()

            record.status = "failed"
            record.stage_key = "failed"
            record.stage_label = "Analysis failed"
            record.completed_at = _utcnow()
            record.error_message = message

    async def run_public_preview_job(
        self,
        *,
        job_id: str,
        image_bytes: bytes,
        filename: str,
        content_type: str,
        crop_hint: str | None,
        inference_service,
        advisory_service,
    ) -> None:
        try:
            self.update_progress(
                job_id,
                stage_key="validating_upload",
                stage_label="Validating image upload",
                progress_percent=5.0,
            )
            validate_upload(
                filename=filename,
                content_type=content_type,
                file_size=len(image_bytes),
            )

            self.update_progress(
                job_id,
                stage_key="starting_analysis",
                stage_label="Preparing analysis workspace",
                progress_percent=10.0,
            )
            preview = await inference_service.predict(
                image_bytes=image_bytes,
                filename=filename,
                content_type=content_type,
                crop_hint=crop_hint,
                progress_callback=self._build_progress_callback(job_id),
            )

            self.update_progress(
                job_id,
                stage_key="generating_advisory",
                stage_label="Generating treatment advice",
                progress_percent=94.0,
            )
            advisory = advisory_service.generate(
                crop=preview.predicted_crop,
                disease=preview.predicted_disease,
                health_status=preview.health_status,
                confidence=preview.confidence_score,
                severity=preview.severity_level,
                urgency=preview.urgency_level,
                analysis_payload=preview.raw_prediction_json,
            )

            self.update_progress(
                job_id,
                stage_key="finalizing",
                stage_label="Packaging results",
                progress_percent=98.0,
            )
            self.complete_job(job_id, result=preview, advisory=advisory)
        except Exception as exc:
            logger.warning("preview_job_failed", extra={"job_id": job_id, "reason": str(exc)})
            self.fail_job(job_id, message=str(exc))

    def _build_progress_callback(self, job_id: str):
        def callback(stage_key: str, progress_percent: float, stage_label: str) -> None:
            self.update_progress(
                job_id,
                stage_key=stage_key,
                stage_label=stage_label,
                progress_percent=progress_percent,
            )

        return callback

    def _snapshot_locked(self, record: _InferenceJobRecord) -> InferenceJobStatusPayload:
        elapsed_seconds = self._elapsed_seconds(record)
        estimated_total_seconds, remaining_seconds = self._estimate_remaining_locked(record, elapsed_seconds)

        return InferenceJobStatusPayload(
            job_id=record.job_id,
            status=record.status,  # type: ignore[arg-type]
            stage_key=record.stage_key,
            stage_label=record.stage_label,
            progress_percent=round(record.progress_percent, 1),
            elapsed_seconds=round(elapsed_seconds, 2),
            estimated_total_seconds=round(estimated_total_seconds, 2) if estimated_total_seconds is not None else None,
            remaining_seconds=round(remaining_seconds, 2) if remaining_seconds is not None else None,
            result=record.result,
            advisory=record.advisory,
            error_message=record.error_message,
            created_at=record.created_at.isoformat(),
            started_at=record.started_at.isoformat() if record.started_at else None,
            completed_at=record.completed_at.isoformat() if record.completed_at else None,
        )

    def _elapsed_seconds(self, record: _InferenceJobRecord) -> float:
        reference = record.completed_at or _utcnow()
        anchor = record.started_at or record.created_at
        return max((reference - anchor).total_seconds(), 0.0)

    def _estimate_remaining_locked(
        self,
        record: _InferenceJobRecord,
        elapsed_seconds: float,
    ) -> tuple[float | None, float | None]:
        if record.status == "completed":
            return elapsed_seconds, 0.0

        if record.progress_percent <= 0 or record.status == "queued":
            if not self._completed_durations:
                return None, None
            average_duration = sum(self._completed_durations) / len(self._completed_durations)
            return average_duration, average_duration

        estimated_from_current = elapsed_seconds / max(record.progress_percent / 100.0, 0.05)
        if self._completed_durations:
            average_duration = sum(self._completed_durations) / len(self._completed_durations)
            estimated_total = average_duration if record.progress_percent < 15 else (estimated_from_current + average_duration) / 2.0
        else:
            estimated_total = estimated_from_current

        remaining_seconds = max(estimated_total - elapsed_seconds, 0.0)
        return estimated_total, remaining_seconds

    def _trim_jobs_locked(self) -> None:
        if len(self._jobs) <= 50:
            return

        finished_jobs = sorted(
            (record for record in self._jobs.values() if record.status in {"completed", "failed"}),
            key=lambda item: item.completed_at or item.created_at,
        )
        while len(self._jobs) > 50 and finished_jobs:
            record = finished_jobs.pop(0)
            self._jobs.pop(record.job_id, None)


_job_service: InferenceJobService | None = None


def get_inference_job_service() -> InferenceJobService:
    global _job_service
    if _job_service is None:
        _job_service = InferenceJobService()
    return _job_service

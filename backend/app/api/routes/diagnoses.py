"""Diagnosis upload and record management routes."""

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Query, UploadFile, status

from app.api.deps import get_db_client
from app.core.exceptions import NotFoundException
from app.core.security import AuthenticatedUser, get_current_user
from app.db.database import fetch_owned_record
from app.schemas.common import ApiResponse, ListResponse
from app.schemas.diagnosis import (
    DiagnosisCreateManual,
    DiagnosisDeleteResponse,
    DiagnosisListItem,
    DiagnosisRead,
    InferenceJobStatusPayload,
    InferencePayload,
)
from app.services.advisory_service import get_advisory_service
from app.services.diagnosis_service import DiagnosisService
from app.services.inference_service import get_inference_service
from app.services.inference_job_service import get_inference_job_service
from app.services.storage_service import get_storage_service
from app.utils.files import validate_upload

router = APIRouter()


def _create_notification_if_needed(db, *, user_id: str, diagnosis: DiagnosisRead) -> None:
    if diagnosis.health_status.value == "healthy":
        return

    try:
        db.table("notifications").insert(
            {
                "user_id": user_id,
                "type": "diagnosis",
                "title": "New plant health alert",
                "message": f"{diagnosis.predicted_crop.title()} scan indicates {diagnosis.predicted_disease or diagnosis.health_status.value}.",
                "related_diagnosis_id": diagnosis.id,
                "related_plant_id": diagnosis.plant_id,
            }
        ).execute()
    except Exception:
        # Notifications should not block saving a diagnosis record.
        return


async def _build_preview_payload_from_bytes(
    *,
    image_bytes: bytes,
    filename: str,
    content_type: str,
    crop_hint: str | None,
    inference_service,
    advisory_service,
) -> tuple[InferencePayload, dict]:
    validate_upload(
        filename=filename,
        content_type=content_type,
        file_size=len(image_bytes),
    )

    preview = await inference_service.predict(
        image_bytes=image_bytes,
        filename=filename,
        content_type=content_type,
        crop_hint=crop_hint,
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
    return preview, {"advisory": advisory.model_dump()}


@router.post("/preview", response_model=ApiResponse[InferencePayload])
async def preview_diagnosis_analysis(
    image: UploadFile = File(...),
    plant_id: str | None = Form(default=None),
    crop_hint: str | None = Form(default=None),
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db_client),
    inference_service=Depends(get_inference_service),
    advisory_service=Depends(get_advisory_service),
) -> ApiResponse[InferencePayload]:
    if plant_id:
        plant = fetch_owned_record(db, "plants", plant_id, current_user.user_id)
        crop_hint = crop_hint or plant.get("crop_type")

    image_bytes = await image.read()
    preview, meta = await _build_preview_payload_from_bytes(
        image_bytes=image_bytes,
        filename=image.filename or "preview-upload.bin",
        content_type=image.content_type or "application/octet-stream",
        crop_hint=crop_hint,
        inference_service=inference_service,
        advisory_service=advisory_service,
    )
    return ApiResponse.success_response(
        data=preview,
        message="Plant computer vision preview generated successfully.",
        meta=meta,
    )


@router.post("/public-preview", response_model=ApiResponse[InferencePayload])
async def public_preview_diagnosis_analysis(
    image: UploadFile = File(...),
    crop_hint: str | None = Form(default=None),
    inference_service=Depends(get_inference_service),
    advisory_service=Depends(get_advisory_service),
) -> ApiResponse[InferencePayload]:
    image_bytes = await image.read()
    preview, meta = await _build_preview_payload_from_bytes(
        image_bytes=image_bytes,
        filename=image.filename or "preview-upload.bin",
        content_type=image.content_type or "application/octet-stream",
        crop_hint=crop_hint,
        inference_service=inference_service,
        advisory_service=advisory_service,
    )
    return ApiResponse.success_response(
        data=preview,
        message="Public plant computer vision preview generated successfully.",
        meta=meta,
    )


@router.post("/public-preview-jobs", response_model=ApiResponse[InferenceJobStatusPayload], status_code=status.HTTP_202_ACCEPTED)
async def start_public_preview_diagnosis_job(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    crop_hint: str | None = Form(default=None),
    inference_service=Depends(get_inference_service),
    advisory_service=Depends(get_advisory_service),
    inference_job_service=Depends(get_inference_job_service),
) -> ApiResponse[InferenceJobStatusPayload]:
    image_bytes = await image.read()
    job = inference_job_service.create_job()

    background_tasks.add_task(
        inference_job_service.run_public_preview_job,
        job_id=job.job_id,
        image_bytes=image_bytes,
        filename=image.filename or "preview-upload.bin",
        content_type=image.content_type or "application/octet-stream",
        crop_hint=crop_hint,
        inference_service=inference_service,
        advisory_service=advisory_service,
    )

    return ApiResponse.success_response(
        data=job,
        message="Public plant preview job queued successfully.",
    )


@router.get("/public-preview-jobs/{job_id}", response_model=ApiResponse[InferenceJobStatusPayload])
def get_public_preview_diagnosis_job(
    job_id: str,
    inference_job_service=Depends(get_inference_job_service),
) -> ApiResponse[InferenceJobStatusPayload]:
    job = inference_job_service.get_job(job_id)
    if not job:
        raise NotFoundException(message="Preview job not found.", code="preview_job_not_found")

    return ApiResponse.success_response(
        data=job,
        message="Public plant preview job fetched successfully.",
    )


@router.post("/upload", response_model=ApiResponse[DiagnosisRead], status_code=status.HTTP_201_CREATED)
async def upload_diagnosis(
    image: UploadFile = File(...),
    plant_id: str | None = Form(default=None),
    farm_notes: str | None = Form(default=None),
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db_client),
    storage_service=Depends(get_storage_service),
    inference_service=Depends(get_inference_service),
    advisory_service=Depends(get_advisory_service),
) -> ApiResponse[DiagnosisRead]:
    crop_hint: str | None = None
    if plant_id:
        plant = fetch_owned_record(db, "plants", plant_id, current_user.user_id)
        crop_hint = plant.get("crop_type")

    service = DiagnosisService(
        db_client=db,
        storage_service=storage_service,
        inference_service=inference_service,
        advisory_service=advisory_service,
    )
    diagnosis = await service.create_from_upload(
        upload=image,
        user=current_user,
        plant_id=plant_id,
        farm_notes=farm_notes,
        crop_hint=crop_hint,
    )
    return ApiResponse.success_response(data=diagnosis, message="Diagnosis created successfully.")


@router.post("", response_model=ApiResponse[DiagnosisRead], status_code=status.HTTP_201_CREATED)
def create_diagnosis_result(
    payload: DiagnosisCreateManual,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db_client),
) -> ApiResponse[DiagnosisRead]:
    if payload.plant_id:
        fetch_owned_record(db, "plants", payload.plant_id, current_user.user_id)

    record = payload.model_dump(exclude_none=True)
    record["user_id"] = current_user.user_id
    result = db.table("diagnoses").insert(record).execute()
    rows = result.data or []
    if not rows:
        raise NotFoundException(message="Diagnosis could not be created.", code="diagnosis_create_failed")

    diagnosis = DiagnosisRead.model_validate(rows[0])
    _create_notification_if_needed(db, user_id=current_user.user_id, diagnosis=diagnosis)
    return ApiResponse.success_response(
        data=diagnosis,
        message="Diagnosis record created successfully.",
    )


@router.get("", response_model=ApiResponse[ListResponse[DiagnosisListItem]])
def list_diagnoses(
    plant_id: str | None = Query(default=None),
    health_status: str | None = Query(default=None),
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db_client),
) -> ApiResponse[ListResponse[DiagnosisListItem]]:
    query = db.table("diagnoses").select("*").eq("user_id", current_user.user_id).order("created_at", desc=True)
    if plant_id:
        query = query.eq("plant_id", plant_id)
    if health_status:
        query = query.eq("health_status", health_status)

    result = query.execute()
    items = [DiagnosisListItem.model_validate(row) for row in (result.data or [])]
    return ApiResponse.success_response(
        data=ListResponse(items=items, total=len(items)),
        message="Diagnoses fetched successfully.",
    )


@router.get("/{diagnosis_id}", response_model=ApiResponse[DiagnosisRead])
def get_diagnosis(
    diagnosis_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db_client),
) -> ApiResponse[DiagnosisRead]:
    record = fetch_owned_record(db, "diagnoses", diagnosis_id, current_user.user_id)
    return ApiResponse.success_response(
        data=DiagnosisRead.model_validate(record),
        message="Diagnosis fetched successfully.",
    )


@router.delete("/{diagnosis_id}", response_model=ApiResponse[DiagnosisDeleteResponse])
def delete_diagnosis(
    diagnosis_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db_client),
    storage_service=Depends(get_storage_service),
) -> ApiResponse[DiagnosisDeleteResponse]:
    record = fetch_owned_record(db, "diagnoses", diagnosis_id, current_user.user_id)
    image_path = record.get("image_path")
    raw_prediction = record.get("raw_prediction_json") or {}
    analysis_run_path = raw_prediction.get("analysis_run_path")

    db.table("diagnoses").delete().eq("id", diagnosis_id).eq("user_id", current_user.user_id).execute()
    if image_path:
        storage_service.delete_file(image_path)
    if analysis_run_path:
        storage_service.delete_file(analysis_run_path)

    return ApiResponse.success_response(
        data=DiagnosisDeleteResponse(id=diagnosis_id, deleted=True),
        message="Diagnosis deleted successfully.",
    )

// Keep the browser on the same loopback address the backend actually binds to.
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1").replace(/\/$/, "");

export type SeverityLevel = "low" | "medium" | "high" | "critical";
export type UrgencyLevel = "low" | "medium" | "high" | "urgent";
export type HealthStatus = "healthy" | "diseased" | "suspicious";
export type PlantStatus = "active" | "monitoring" | "diseased" | "archived";
export type TreatmentStatus = "planned" | "applied" | "monitoring" | "completed";
export type NotificationType = "diagnosis" | "follow_up" | "treatment" | "system";

export type ApiResponse<T> = {
  success: boolean;
  message: string;
  data: T;
  meta?: Record<string, unknown> | null;
};

export type ListResponse<T> = {
  items: T[];
  total: number;
};

export type SessionTokens = {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in?: number | null;
  expires_at?: number | null;
};

export type UserSessionData = {
  user_id: string;
  email: string;
  email_confirmed_at?: string | null;
  session?: SessionTokens | null;
};

export type ProfilePayload = {
  id: string;
  full_name?: string | null;
  email: string;
  farm_name?: string | null;
  phone?: string | null;
  location?: string | null;
  avatar_url?: string | null;
  created_at: string;
  updated_at: string;
};

export type AdvisoryPayload = {
  advisory_text: string;
  advisory_source: string;
  llm_model?: string | null;
  what_happened: string;
  likely_cause: string;
  severity_summary: string;
  why_it_matters: string;
  treatment_steps: string[];
  action_explanations: string[];
  prevention_tips: string[];
  urgency_guidance: string;
  follow_up_recommendation: string;
};

export type DiagnosisPreviewPayload = {
  predicted_crop: string;
  health_status: HealthStatus;
  predicted_disease?: string | null;
  confidence_score: number;
  severity_level: SeverityLevel;
  urgency_level: UrgencyLevel;
  model_version: string;
  raw_prediction_json: Record<string, unknown>;
};

export type DiagnosisPreviewResponse = {
  success: boolean;
  message: string;
  data: DiagnosisPreviewPayload;
  meta?: {
    advisory?: AdvisoryPayload;
  };
};

export type DiagnosisPreviewJobStatus = "queued" | "running" | "completed" | "failed";

export type DiagnosisPreviewJobPayload = {
  job_id: string;
  status: DiagnosisPreviewJobStatus;
  stage_key: string;
  stage_label: string;
  progress_percent: number;
  elapsed_seconds: number;
  estimated_total_seconds?: number | null;
  remaining_seconds?: number | null;
  result?: DiagnosisPreviewPayload | null;
  advisory?: AdvisoryPayload | null;
  error_message?: string | null;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
};

export type DiagnosisRecord = DiagnosisPreviewPayload & {
  id: string;
  user_id: string;
  plant_id?: string | null;
  image_url: string;
  image_path: string;
  ai_advice_text: string;
  advisory_payload: Record<string, unknown>;
  created_at: string;
};

export type PlantRecord = {
  id: string;
  user_id: string;
  farm_id?: string | null;
  crop_type: string;
  custom_name: string;
  zone_or_field?: string | null;
  planted_at?: string | null;
  status: PlantStatus;
  notes?: string | null;
  created_at: string;
  updated_at: string;
};

export type TreatmentLogRecord = {
  id: string;
  user_id: string;
  plant_id: string;
  diagnosis_id?: string | null;
  treatment_action: string;
  notes?: string | null;
  applied_at: string;
  follow_up_date?: string | null;
  status: TreatmentStatus;
  created_at: string;
};

export type NotificationRecord = {
  id: string;
  user_id: string;
  type: NotificationType;
  title: string;
  message: string;
  read: boolean;
  related_diagnosis_id?: string | null;
  related_plant_id?: string | null;
  created_at: string;
};

export type HealthDistributionItem = {
  health_status: string;
  count: number;
};

export type DashboardSummaryPayload = {
  total_plants: number;
  total_diagnoses: number;
  recent_diagnoses: DiagnosisRecord[];
  active_alerts: number;
  treatment_follow_ups_due: number;
  crop_health_distribution: HealthDistributionItem[];
};

export type UserSettingsPayload = {
  user_id: string;
  disease_detection_alerts: boolean;
  treatment_reminders: boolean;
  weekly_health_reports: boolean;
  ai_tips_recommendations: boolean;
  auto_generate_treatment_plans: boolean;
  seasonal_recommendations: boolean;
  detailed_analysis_mode: boolean;
  share_anonymized_data: boolean;
  keep_detection_history: boolean;
  auto_detect_crop_type: boolean;
  auto_save_scans: boolean;
  created_at: string;
  updated_at: string;
};

export type AdvisorReplyPayload = {
  reply: string;
  source: string;
  llm_model?: string | null;
  context_summary: string;
  context_images?: {
    title: string;
    url: string;
    source: string;
  }[];
};

export type AdvisorContextPayload = {
  current_diagnosis?: Record<string, unknown> | null;
  context_mode?: "account" | "current_scan";
  model_mode?: "fast" | "thinking";
};

function extractErrorMessage(payload: unknown, fallback: string): string {
  if (!payload || typeof payload !== "object") {
    return fallback;
  }

  if ("message" in payload && typeof payload.message === "string") {
    return payload.message;
  }

  if ("error" in payload && typeof payload.error === "object" && payload.error !== null) {
    const nested = payload.error as Record<string, unknown>;
    if (typeof nested.message === "string") {
      return nested.message;
    }
  }

  return fallback;
}

export async function previewDiagnosis(
  file: File,
  options?: { cropHint?: string | null },
): Promise<DiagnosisPreviewResponse> {
  const formData = new FormData();
  formData.append("image", file);
  if (options?.cropHint) {
    formData.append("crop_hint", options.cropHint);
  }

  const response = await fetch(`${API_BASE_URL}/diagnoses/public-preview`, {
    method: "POST",
    body: formData,
  });

  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok) {
    throw new Error(extractErrorMessage(payload, "Plant analysis failed. Please try another image."));
  }

  return payload as DiagnosisPreviewResponse;
}

export async function startPreviewDiagnosisJob(
  file: File,
  options?: { cropHint?: string | null },
): Promise<ApiResponse<DiagnosisPreviewJobPayload>> {
  const formData = new FormData();
  formData.append("image", file);
  if (options?.cropHint) {
    formData.append("crop_hint", options.cropHint);
  }

  const response = await fetch(`${API_BASE_URL}/diagnoses/public-preview-jobs`, {
    method: "POST",
    body: formData,
  });

  return parseJsonResponse<ApiResponse<DiagnosisPreviewJobPayload>>(
    response,
    "Could not start the plant analysis job.",
  );
}

export async function getPreviewDiagnosisJob(
  jobId: string,
): Promise<ApiResponse<DiagnosisPreviewJobPayload>> {
  return apiRequest<ApiResponse<DiagnosisPreviewJobPayload>>(`/diagnoses/public-preview-jobs/${jobId}`, {
    method: "GET",
    fallback: "Could not fetch the plant analysis progress.",
  });
}

async function parseJsonResponse<T>(response: Response, fallback: string): Promise<T> {
  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok) {
    throw new Error(extractErrorMessage(payload, fallback));
  }

  return payload as T;
}

async function apiRequest<T>(
  path: string,
  options: RequestInit & { token?: string | null; fallback?: string } = {},
): Promise<T> {
  const { token, fallback, headers, ...requestOptions } = options;
  const nextHeaders = new Headers(headers);
  if (token) {
    nextHeaders.set("Authorization", `Bearer ${token}`);
  }
  if (requestOptions.body && !(requestOptions.body instanceof FormData) && !nextHeaders.has("Content-Type")) {
    nextHeaders.set("Content-Type", "application/json");
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...requestOptions,
    headers: nextHeaders,
  });
  return parseJsonResponse<T>(response, fallback ?? "Request failed.");
}

export async function loginWithPassword(email: string, password: string): Promise<ApiResponse<UserSessionData>> {
  return apiRequest<ApiResponse<UserSessionData>>("/auth/log-in", {
    method: "POST",
    body: JSON.stringify({ email, password }),
    fallback: "Login failed. Check your email and password.",
  });
}

export async function signUpWithPassword(payload: {
  email: string;
  password: string;
  full_name: string;
  farm_name?: string | null;
  phone?: string | null;
}): Promise<ApiResponse<UserSessionData>> {
  return apiRequest<ApiResponse<UserSessionData>>("/auth/sign-up", {
    method: "POST",
    body: JSON.stringify(payload),
    fallback: "Sign up failed. Please check your details.",
  });
}

export async function fetchCurrentProfile(token: string): Promise<ApiResponse<ProfilePayload>> {
  return apiRequest<ApiResponse<ProfilePayload>>("/auth/me", {
    method: "GET",
    token,
    fallback: "Could not fetch your profile.",
  });
}

export async function logoutSession(token: string): Promise<ApiResponse<{ logged_out: boolean }>> {
  return apiRequest<ApiResponse<{ logged_out: boolean }>>("/auth/log-out", {
    method: "POST",
    token,
    fallback: "Logout failed.",
  });
}

function readRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

function readString(value: unknown): string | null {
  return typeof value === "string" && value.trim().length > 0 ? value : null;
}

export function canSavePreview(result: DiagnosisPreviewResponse | null): boolean {
  if (!result) {
    return false;
  }
  const raw = readRecord(result.data.raw_prediction_json);
  const input = readRecord(raw.input);
  return Boolean(readString(input.stored_input_url) && readString(input.stored_input_path));
}

export async function saveDiagnosisPreview(
  result: DiagnosisPreviewResponse,
  token: string,
  options?: { plantId?: string | null },
): Promise<ApiResponse<DiagnosisRecord>> {
  const raw = readRecord(result.data.raw_prediction_json);
  const input = readRecord(raw.input);
  const imageUrl = readString(input.stored_input_url);
  const imagePath = readString(input.stored_input_path);
  const advisory = result.meta?.advisory;

  if (!imageUrl || !imagePath) {
    throw new Error("This preview cannot be saved because the stored image path is missing.");
  }

  return apiRequest<ApiResponse<DiagnosisRecord>>("/diagnoses", {
    method: "POST",
    token,
    body: JSON.stringify({
      plant_id: options?.plantId ?? null,
      image_url: imageUrl,
      image_path: imagePath,
      predicted_crop: result.data.predicted_crop,
      health_status: result.data.health_status,
      predicted_disease: result.data.predicted_disease ?? null,
      confidence_score: result.data.confidence_score,
      severity_level: result.data.severity_level,
      urgency_level: result.data.urgency_level,
      model_version: result.data.model_version,
      raw_prediction_json: result.data.raw_prediction_json,
      ai_advice_text: advisory?.advisory_text ?? advisory?.what_happened ?? "No advisory text was generated.",
      advisory_payload: advisory ?? {},
    }),
    fallback: "Could not save this diagnosis.",
  });
}

export async function listDiagnoses(
  token: string,
  options?: { plantId?: string | null; healthStatus?: HealthStatus | null },
): Promise<ApiResponse<ListResponse<DiagnosisRecord>>> {
  const params = new URLSearchParams();
  if (options?.plantId) {
    params.set("plant_id", options.plantId);
  }
  if (options?.healthStatus) {
    params.set("health_status", options.healthStatus);
  }
  const suffix = params.toString() ? `?${params.toString()}` : "";
  return apiRequest<ApiResponse<ListResponse<DiagnosisRecord>>>(`/diagnoses${suffix}`, {
    method: "GET",
    token,
    fallback: "Could not load diagnosis history.",
  });
}

export async function deleteDiagnosis(id: string, token: string): Promise<ApiResponse<{ id: string; deleted: boolean }>> {
  return apiRequest<ApiResponse<{ id: string; deleted: boolean }>>(`/diagnoses/${id}`, {
    method: "DELETE",
    token,
    fallback: "Could not delete this diagnosis.",
  });
}

export async function updateProfile(
  token: string,
  payload: {
    full_name?: string | null;
    farm_name?: string | null;
    phone?: string | null;
    location?: string | null;
    avatar_url?: string | null;
  },
): Promise<ApiResponse<ProfilePayload>> {
  return apiRequest<ApiResponse<ProfilePayload>>("/profile", {
    method: "PATCH",
    token,
    body: JSON.stringify(payload),
    fallback: "Could not update your profile.",
  });
}

export async function requestPasswordReset(email: string): Promise<ApiResponse<{ recorded: boolean; delivery_mode: string; note: string }>> {
  return apiRequest<ApiResponse<{ recorded: boolean; delivery_mode: string; note: string }>>("/auth/request-password-reset", {
    method: "POST",
    body: JSON.stringify({ email }),
    fallback: "Could not submit the password reset request.",
  });
}

export async function changePassword(
  token: string,
  payload: { current_password: string; new_password: string },
): Promise<ApiResponse<{ updated: boolean }>> {
  return apiRequest<ApiResponse<{ updated: boolean }>>("/auth/change-password", {
    method: "POST",
    token,
    body: JSON.stringify(payload),
    fallback: "Could not update your password.",
  });
}

export async function fetchDashboardSummary(token: string): Promise<ApiResponse<DashboardSummaryPayload>> {
  return apiRequest<ApiResponse<DashboardSummaryPayload>>("/dashboard/summary", {
    method: "GET",
    token,
    fallback: "Could not load the dashboard.",
  });
}

export async function listPlants(
  token: string,
  options?: { cropType?: string | null; status?: PlantStatus | null; includeArchived?: boolean },
): Promise<ApiResponse<ListResponse<PlantRecord>>> {
  const params = new URLSearchParams();
  if (options?.cropType) {
    params.set("crop_type", options.cropType);
  }
  if (options?.status) {
    params.set("status", options.status);
  }
  if (options?.includeArchived) {
    params.set("include_archived", "true");
  }
  const suffix = params.toString() ? `?${params.toString()}` : "";
  return apiRequest<ApiResponse<ListResponse<PlantRecord>>>(`/plants${suffix}`, {
    method: "GET",
    token,
    fallback: "Could not load your plants.",
  });
}

export async function fetchPlant(id: string, token: string): Promise<ApiResponse<PlantRecord>> {
  return apiRequest<ApiResponse<PlantRecord>>(`/plants/${id}`, {
    method: "GET",
    token,
    fallback: "Could not load this plant.",
  });
}

export async function createPlant(
  token: string,
  payload: {
    crop_type: string;
    custom_name: string;
    zone_or_field?: string | null;
    planted_at?: string | null;
    status?: PlantStatus;
    notes?: string | null;
    farm_id?: string | null;
  },
): Promise<ApiResponse<PlantRecord>> {
  return apiRequest<ApiResponse<PlantRecord>>("/plants", {
    method: "POST",
    token,
    body: JSON.stringify(payload),
    fallback: "Could not create the plant.",
  });
}

export async function updatePlant(
  id: string,
  token: string,
  payload: {
    crop_type?: string | null;
    custom_name?: string | null;
    zone_or_field?: string | null;
    planted_at?: string | null;
    status?: PlantStatus | null;
    notes?: string | null;
    farm_id?: string | null;
  },
): Promise<ApiResponse<PlantRecord>> {
  return apiRequest<ApiResponse<PlantRecord>>(`/plants/${id}`, {
    method: "PATCH",
    token,
    body: JSON.stringify(payload),
    fallback: "Could not update the plant.",
  });
}

export async function archivePlant(id: string, token: string): Promise<ApiResponse<PlantRecord>> {
  return apiRequest<ApiResponse<PlantRecord>>(`/plants/${id}`, {
    method: "DELETE",
    token,
    fallback: "Could not archive the plant.",
  });
}

export async function listTreatmentLogs(
  token: string,
  options?: { plantId?: string | null; status?: TreatmentStatus | null },
): Promise<ApiResponse<ListResponse<TreatmentLogRecord>>> {
  const params = new URLSearchParams();
  if (options?.plantId) {
    params.set("plant_id", options.plantId);
  }
  if (options?.status) {
    params.set("status", options.status);
  }
  const suffix = params.toString() ? `?${params.toString()}` : "";
  return apiRequest<ApiResponse<ListResponse<TreatmentLogRecord>>>(`/treatment-logs${suffix}`, {
    method: "GET",
    token,
    fallback: "Could not load treatment logs.",
  });
}

export async function createTreatmentLog(
  token: string,
  payload: {
    plant_id: string;
    diagnosis_id?: string | null;
    treatment_action: string;
    notes?: string | null;
    applied_at: string;
    follow_up_date?: string | null;
    status?: TreatmentStatus;
  },
): Promise<ApiResponse<TreatmentLogRecord>> {
  return apiRequest<ApiResponse<TreatmentLogRecord>>("/treatment-logs", {
    method: "POST",
    token,
    body: JSON.stringify(payload),
    fallback: "Could not create the treatment log.",
  });
}

export async function updateTreatmentLog(
  id: string,
  token: string,
  payload: {
    plant_id?: string | null;
    diagnosis_id?: string | null;
    treatment_action?: string | null;
    notes?: string | null;
    applied_at?: string | null;
    follow_up_date?: string | null;
    status?: TreatmentStatus | null;
  },
): Promise<ApiResponse<TreatmentLogRecord>> {
  return apiRequest<ApiResponse<TreatmentLogRecord>>(`/treatment-logs/${id}`, {
    method: "PATCH",
    token,
    body: JSON.stringify(payload),
    fallback: "Could not update the treatment log.",
  });
}

export async function deleteTreatmentLog(id: string, token: string): Promise<ApiResponse<{ id: string; deleted: boolean }>> {
  return apiRequest<ApiResponse<{ id: string; deleted: boolean }>>(`/treatment-logs/${id}`, {
    method: "DELETE",
    token,
    fallback: "Could not delete the treatment log.",
  });
}

export async function listNotifications(token: string): Promise<ApiResponse<ListResponse<NotificationRecord>>> {
  return apiRequest<ApiResponse<ListResponse<NotificationRecord>>>("/notifications", {
    method: "GET",
    token,
    fallback: "Could not load notifications.",
  });
}

export async function markNotificationRead(id: string, token: string): Promise<ApiResponse<NotificationRecord>> {
  return apiRequest<ApiResponse<NotificationRecord>>(`/notifications/${id}/read`, {
    method: "POST",
    token,
    fallback: "Could not mark the notification as read.",
  });
}

export async function markAllNotificationsRead(token: string): Promise<ApiResponse<{ updated_count: number }>> {
  return apiRequest<ApiResponse<{ updated_count: number }>>("/notifications/read-all", {
    method: "POST",
    token,
    fallback: "Could not mark all notifications as read.",
  });
}

export async function fetchSettings(token: string): Promise<ApiResponse<UserSettingsPayload>> {
  return apiRequest<ApiResponse<UserSettingsPayload>>("/settings", {
    method: "GET",
    token,
    fallback: "Could not load your settings.",
  });
}

export async function updateSettings(
  token: string,
  payload: Partial<Omit<UserSettingsPayload, "user_id" | "created_at" | "updated_at">>,
): Promise<ApiResponse<UserSettingsPayload>> {
  return apiRequest<ApiResponse<UserSettingsPayload>>("/settings", {
    method: "PATCH",
    token,
    body: JSON.stringify(payload),
    fallback: "Could not update your settings.",
  });
}

export async function askAdvisor(
  token: string,
  message: string,
  context?: AdvisorContextPayload,
): Promise<ApiResponse<AdvisorReplyPayload>> {
  return apiRequest<ApiResponse<AdvisorReplyPayload>>("/advisor/chat", {
    method: "POST",
    token,
    body: JSON.stringify({
      message,
      context_mode: context?.context_mode ?? (context?.current_diagnosis ? "current_scan" : "account"),
      model_mode: context?.model_mode ?? "fast",
      current_diagnosis: context?.current_diagnosis ?? null,
    }),
    fallback: "Could not get an advisor reply right now.",
  });
}

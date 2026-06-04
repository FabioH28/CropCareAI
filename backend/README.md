# CropCare AI Backend

FastAPI backend for CropCare AI using a local MySQL or MariaDB database, local JWT auth, and filesystem-based image storage.

## Stack

- FastAPI
- MySQL or MariaDB via XAMPP
- Local JWT authentication
- Local file uploads served from `/uploads`
- Plant computer-vision analysis delegated to `ml/plant/cv/analyze.py`

## Folder Structure

```text
repository/
  backend/
    app/
      api/
      core/
      db/
      models/
      schemas/
      services/
      utils/
      main.py
    tests/
    .env.example
    requirements.txt
  database/
    migrations/
      001_initial_schema.sql
```

## Environment Variables

Copy `.env.example` to `.env` and update the values as needed.

Required local values:

- `DATABASE_URL`
- `AUTH_SECRET_KEY`

Useful defaults already included:

- `DATABASE_URL=mysql+pymysql://root:@127.0.0.1:3308/cropcare_ai`
- `BACKEND_PUBLIC_URL=http://127.0.0.1:8000`
- `UPLOADS_ROOT=storage`
- `CORS_ALLOW_ORIGINS=["http://127.0.0.1:8080","http://localhost:8080","http://localhost:5173"]`
- `DL_CHECKPOINT_PATH=ml/plant/artifacts/production/best_model.pt`
- `DL_TTA_PASSES=1`
- `LLM_BASE_URL=http://localhost:11434`
- `LLM_MODEL=llama3:8b`

## XAMPP Setup

1. Start the project database with `../scripts/start-cropcare-db.ps1`.
2. The dedicated MariaDB instance uses `../database/xampp-local/my.ini` and listens on `127.0.0.1:3308`.
3. Run:
   - [../database/migrations/001_initial_schema.sql](../database/migrations/001_initial_schema.sql)
   - [../database/migrations/002_app_support.sql](../database/migrations/002_app_support.sql)
   - [../database/seeds/001_fabio_seed.sql](../database/seeds/001_fabio_seed.sql)
   - [../database/seeds/002_fabio_app_seed.sql](../database/seeds/002_fabio_app_seed.sql)
4. Copy `.env.example` to `.env`.
5. Keep `DATABASE_URL` aligned with the project-local MariaDB instance unless you intentionally move the database to another port or password.

## Backend Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Uploaded diagnosis images are stored under `backend/storage/diagnosis-images/...` by default and served at `http://localhost:8000/uploads/...`.

For the full plant analysis pipeline, also install the ML environment in `ml/plant/`:

```bash
cd ..\ml\plant
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you want short natural-language explanations, keep Ollama running locally with the configured model:

```bash
ollama run llama3:8b
```

## Auth Behavior

- `POST /api/v1/auth/sign-up` creates a local user record and profile.
- `POST /api/v1/auth/log-in` verifies the local password hash and returns JWTs.
- Protected endpoints expect `Authorization: Bearer <access_token>`.
- `POST /api/v1/auth/log-out` is stateless and simply tells the client to discard the token.

## Route Summary

### System

- `GET /`
- `GET /healthz`

### Auth

- `POST /api/v1/auth/sign-up`
- `POST /api/v1/auth/log-in`
- `POST /api/v1/auth/log-out`
- `GET /api/v1/auth/me`

### Profile

- `GET /api/v1/profile`
- `PATCH /api/v1/profile`

### Plants

- `POST /api/v1/plants`
- `GET /api/v1/plants`
- `GET /api/v1/plants/{plant_id}`
- `PATCH /api/v1/plants/{plant_id}`
- `DELETE /api/v1/plants/{plant_id}`

### Diagnoses

- `POST /api/v1/diagnoses/preview`
- `POST /api/v1/diagnoses/upload`
- `POST /api/v1/diagnoses`
- `GET /api/v1/diagnoses`
- `GET /api/v1/diagnoses/{diagnosis_id}`
- `DELETE /api/v1/diagnoses/{diagnosis_id}`

### Treatment Logs

- `POST /api/v1/treatment-logs`
- `GET /api/v1/treatment-logs`
- `GET /api/v1/treatment-logs/{treatment_log_id}`
- `PATCH /api/v1/treatment-logs/{treatment_log_id}`
- `DELETE /api/v1/treatment-logs/{treatment_log_id}`

### Notifications

- `GET /api/v1/notifications`
- `POST /api/v1/notifications/{notification_id}/read`
- `POST /api/v1/notifications/read-all`

### Dashboard

- `GET /api/v1/dashboard/summary`

## Example Sign Up Response

```json
{
  "success": true,
  "message": "Account created successfully.",
  "data": {
    "user_id": "user-uuid",
    "email": "farmer@example.com",
    "email_confirmed_at": "2026-04-20T12:00:00+00:00",
    "session": {
      "access_token": "jwt",
      "refresh_token": "refresh-jwt",
      "token_type": "bearer",
      "expires_in": 3600,
      "expires_at": 1770000000
    }
  }
}
```

## Example CV Preview Response

```json
{
  "success": true,
  "message": "Plant computer vision preview generated successfully.",
  "data": {
    "predicted_crop": "apple",
    "health_status": "suspicious",
    "predicted_disease": "chlorosis_or_nutrient_stress_pattern",
    "confidence_score": 0.696,
    "severity_level": "medium",
    "urgency_level": "medium",
    "model_version": "plant-vision-cv-v2-leaf-first",
    "raw_prediction_json": {
      "pipeline": "plant-vision-cv-v2-leaf-first",
      "topics_covered": [
        "image_restoration_and_reconstruction",
        "color_image_processing",
        "segmentation_for_leaf_and_lesion_isolation",
        "graph_cut_segmentation",
        "feature_extraction_and_pattern_classification"
      ],
      "supplementary_topics_demonstrated": [
        "frequency_domain_analysis_for_interpretability",
        "wavelet_analysis_for_texture_comparison"
      ],
      "outputs": [
        {
          "stage": "frequency",
          "label": "fft_magnitude_spectrum",
          "image_url": "http://localhost:8000/uploads/analysis-runs/run-id/09_frequency_fft-magnitude-spectrum.png"
        }
      ],
      "classification": {
        "baseline": {
          "method": "baseline_thresholding",
          "infected_area_percentage": 41.71
        },
        "improved": {
          "method": "consensus_multistage_cv",
          "infected_area_percentage": 9.53
        }
      },
      "severity_estimation": {
        "infected_area_percentage": 9.53
      }
    }
  },
  "meta": {
    "advisory": {
      "advisory_source": "ollama",
      "what_happened": "The leaf shows early suspicious stress signs.",
      "likely_cause": "It may be nutrient stress or an early disease pattern.",
      "severity_summary": "Severity looks medium.",
      "treatment_steps": [
        "Inspect nearby leaves.",
        "Correct the likely stress source.",
        "Monitor the plant over the next 48 to 72 hours."
      ]
    }
  }
}
```

## Example Upload Response

```json
{
  "success": true,
  "message": "Diagnosis created successfully.",
  "data": {
    "id": "diagnosis-uuid",
    "user_id": "user-uuid",
    "plant_id": "plant-uuid",
    "image_url": "http://localhost:8000/uploads/diagnosis-images/user-id/path/file.jpg",
    "image_path": "diagnosis-images/user-id/path/file.jpg",
    "predicted_crop": "apple",
    "health_status": "suspicious",
    "predicted_disease": "chlorosis_or_nutrient_stress_pattern",
    "confidence_score": 0.696,
    "severity_level": "medium",
    "urgency_level": "medium",
    "model_version": "plant-vision-cv-v2-leaf-first",
    "raw_prediction_json": {
      "pipeline": "plant-vision-cv-v2-leaf-first",
      "summary": {
        "infected_area_percentage": 9.53
      }
    },
    "ai_advice_text": "The model suggests apple is suspicious with 70% confidence.",
    "advisory_payload": {
      "advisory_text": "The model suggests tomato is diseased with 91% confidence.",
      "treatment_steps": [
        "Remove visibly infected leaves or plant material where safe to do so.",
        "Inspect nearby plants for similar symptoms.",
        "Isolate or tag the affected tomato area for close monitoring.",
        "Apply crop-safe treatment after confirming local agronomy guidance."
      ],
      "prevention_tips": [
        "Avoid overhead irrigation if fungal spread is suspected.",
        "Improve airflow and spacing around the crop canopy.",
        "Keep tools clean between zones to reduce cross-contamination."
      ],
      "urgency_guidance": "Recommended urgency: urgent. Prioritize follow-up based on this level.",
      "follow_up_recommendation": "Re-scan the plant in 48-72 hours after treatment or sooner if symptoms worsen."
    },
    "created_at": "2026-04-16T10:00:00+00:00"
  }
}
```

## Plant CV Coverage

- preprocessing and image manipulation
- enhancement and restoration
- color analysis and leaf-first ROI isolation
- deployed lesion segmentation with thresholding, clustering, and graph cut
- feature extraction
- healthy vs diseased classification
- disease severity and infected-area estimation
- baseline-vs-improved comparison
- supplementary FFT, wavelet, region-growing, split-merge, and superpixel outputs for comparison and interpretability

## LLM Explanation Layer

- local Ollama-backed explanation service for short human summaries
- default configured model: `llama3:8b`
- used to generate:
  - what happened
  - likely cause
  - severity
  - what to do now
  - why the actions help

## Learned Model Integration

The backend now calls both:

- the plant CV pipeline for preprocessing, segmentation, infected-area estimation, and severity
- the production deep-learning checkpoint at `ml/plant/artifacts/production/best_model.pt` for final crop and disease classification

The merge happens in [app/services/inference_service.py](app/services/inference_service.py), and the model source is included in `raw_prediction_json.model_integration`.

## Testing

```bash
pytest
```

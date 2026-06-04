# System Overview

## What CropCare AI Is

CropCare AI is a plant-only smart farming application. It combines:

- a React frontend
- a FastAPI backend
- a local MariaDB/XAMPP database
- a plant computer vision pipeline
- a deep learning disease classifier
- a short explanation layer

The project is now fully wired as an app without changing the selected production plant model.

## Main Runtime Flow

1. The user signs in through the frontend.
2. The frontend sends requests to the FastAPI backend.
3. The backend stores users, plants, diagnoses, treatments, notifications, and settings in the local MariaDB database.
4. On diagnosis, the backend runs the plant CV pipeline and then the selected production deep model.
5. The backend returns the diagnosis, severity, infected-area estimate, and short advisory text.
6. The app shows the result and can save it to history.

## What Is Real In The App

The following app areas are now backed by the real backend and database:

- authentication
- dashboard
- diagnose
- saved diagnosis history
- plant management
- plant detail pages
- treatment log
- notifications
- profile
- settings
- forgot-password request recording
- advisor chat

## Project Structure

- `frontend/`
  - React app and pages

- `backend/`
  - FastAPI routes, auth, storage, orchestration, and API services

- `ml/plant/cv/`
  - plant computer vision and digital image processing

- `ml/plant/dl/`
  - deep learning training and inference

- `ml/plant/llm/`
  - explanation helpers used by the advisory layer

- `database/`
  - SQL migrations, seeds, and local XAMPP DB config

- `datasets/plant/`
  - plant datasets used for experiments and training

## Database Setup

The project uses a dedicated local MariaDB instance on:

- host: `127.0.0.1`
- port: `3308`
- database: `cropcare_ai`

Important files:

- `database/migrations/001_initial_schema.sql`
- `database/migrations/002_app_support.sql`
- `database/seeds/001_fabio_seed.sql`
- `database/seeds/002_fabio_app_seed.sql`
- `database/xampp-local/my.ini`

## Main Seeded Account

The project includes one seeded account for testing:

- email: `fabio@cropcareai.app`
- password: `Fabio123!`

This seed creates the Fabio user, profile, farm record, and settings. It does not prefill plant diagnoses, treatment logs, or notifications.

## Startup Scripts

Use these scripts to run the stack:

- `scripts/start-cropcare-db.ps1`
- `scripts/run-backend.ps1`
- `scripts/run-frontend.ps1`

## Core Code Path

If you only want the real diagnosis path, read these files:

1. `frontend/src/pages/DiagnosePage.tsx`
2. `backend/app/api/routes/diagnoses.py`
3. `backend/app/services/inference_service.py`
4. `ml/plant/cv/analysis_pipeline.py`
5. `ml/plant/dl/infer.py`
6. `backend/app/services/advisory_service.py`

## What The Current Model Can Predict

The production plant classifier supports:

- `6` crop groups
- `27` total class labels
- `21` diseased labels
- `6` healthy labels

Supported crops:

- apple
- corn
- grape
- pepper
- potato
- tomato

Important rule:

- the model only names a disease when the leaf belongs to one of those supported crops
- if the leaf is outside the supported crop set, the system now returns `unknown_leaf_crop` instead of forcing a wrong crop label

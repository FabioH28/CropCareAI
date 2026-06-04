# Database

This folder contains database setup assets for the backend.

## Migrations

- `migrations/001_initial_schema.sql`: MySQL schema for the local XAMPP database. Creates the users, profiles, plants, diagnoses, treatment logs, and notifications tables.

## Project-local XAMPP instance

- `xampp-local/my.ini`: dedicated MariaDB configuration for a project-local instance that runs on port `3308`

This avoids depending on whatever other MySQL service may already be using `3306` on the machine.

## Seeds

- `seeds/001_fabio_seed.sql`: Fabio user / profile / farm seed for the local CropCare database

## Run Scripts

- `../scripts/start-cropcare-db.ps1`: starts the project-local MariaDB instance on port `3308`
- `../scripts/run-backend.ps1`: starts the FastAPI backend on `127.0.0.1:8000`
- `../scripts/run-frontend.ps1`: starts the Vite frontend on `127.0.0.1:8080`

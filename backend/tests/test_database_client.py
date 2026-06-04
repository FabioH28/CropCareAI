from pathlib import Path

from app.db.database import DatabaseClient


def test_database_client_supports_basic_crud_and_json(tmp_path: Path):
    database_path = tmp_path / "cropcare_test.db"
    client = DatabaseClient.connect(f"sqlite:///{database_path}")

    try:
        client.connection.executescript(
            """
            CREATE TABLE users (
              id TEXT PRIMARY KEY,
              email TEXT NOT NULL UNIQUE,
              password_hash TEXT NOT NULL,
              email_confirmed_at TEXT,
              is_active INTEGER NOT NULL DEFAULT 1,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP,
              updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE diagnoses (
              id TEXT PRIMARY KEY,
              user_id TEXT NOT NULL,
              image_url TEXT NOT NULL,
              image_path TEXT NOT NULL,
              predicted_crop TEXT NOT NULL,
              health_status TEXT NOT NULL,
              predicted_disease TEXT,
              confidence_score REAL NOT NULL,
              severity_level TEXT NOT NULL,
              urgency_level TEXT NOT NULL,
              model_version TEXT NOT NULL,
              raw_prediction_json TEXT NOT NULL,
              ai_advice_text TEXT NOT NULL,
              advisory_payload TEXT NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        client.commit()

        user = client.table("users").insert(
            {
                "id": "user-1",
                "email": "farmer@example.com",
                "password_hash": "hashed",
                "email_confirmed_at": "2026-04-20T10:00:00+00:00",
                "is_active": True,
            }
        ).execute().data[0]
        assert user["email"] == "farmer@example.com"

        diagnosis = client.table("diagnoses").insert(
            {
                "id": "diag-1",
                "user_id": "user-1",
                "image_url": "http://localhost:8000/uploads/diagnosis-images/user-1/file.jpg",
                "image_path": "diagnosis-images/user-1/file.jpg",
                "predicted_crop": "tomato",
                "health_status": "diseased",
                "predicted_disease": "early_blight",
                "confidence_score": 0.91,
                "severity_level": "high",
                "urgency_level": "urgent",
                "model_version": "mock-cropcare-v1",
                "raw_prediction_json": {"pipeline": "mock"},
                "ai_advice_text": "Test advice",
                "advisory_payload": {"steps": ["remove infected leaves"]},
            }
        ).execute().data[0]
        assert diagnosis["raw_prediction_json"]["pipeline"] == "mock"

        updated = client.table("users").update({"is_active": False}).eq("id", "user-1").execute().data[0]
        assert updated["is_active"] is False

        selected = client.table("diagnoses").select("*").eq("user_id", "user-1").limit(1).execute().data
        assert len(selected) == 1

        deleted = client.table("diagnoses").delete().eq("id", "diag-1").execute().data
        assert deleted[0]["id"] == "diag-1"
        assert client.table("diagnoses").select("*").eq("id", "diag-1").execute().data == []
    finally:
        client.close()

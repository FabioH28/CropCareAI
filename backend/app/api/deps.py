"""Reusable FastAPI dependencies."""

from collections.abc import Generator

from app.db.database import DatabaseClient


def get_db_client() -> Generator:
    """Expose the configured project database client as a dependency."""

    client = DatabaseClient.connect()
    try:
        yield client
    finally:
        client.close()

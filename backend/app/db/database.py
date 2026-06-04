"""Local database adapter with a small query-builder API."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

from app.core.config import get_settings
from app.core.exceptions import AppException, NotFoundException

settings = get_settings()
JSON_COLUMNS: dict[str, set[str]] = {
    "diagnoses": {"raw_prediction_json", "advisory_payload"},
}
UPDATED_AT_TABLES = {"users", "profiles", "farms", "plants"}
PRIMARY_KEY_COLUMNS = {
    "user_preferences": "user_id",
}
BOOLEAN_COLUMNS = {
    "read",
    "is_active",
    "disease_detection_alerts",
    "treatment_reminders",
    "weekly_health_reports",
    "ai_tips_recommendations",
    "auto_generate_treatment_plans",
    "seasonal_recommendations",
    "detailed_analysis_mode",
    "share_anonymized_data",
    "keep_detection_history",
    "auto_detect_crop_type",
    "auto_save_scans",
}
RESOURCE_NOT_FOUND_META = {
    "farms": ("Farm not found.", "farm_not_found"),
    "plants": ("Plant not found.", "plant_not_found"),
    "diagnoses": ("Diagnosis not found.", "diagnosis_not_found"),
    "treatment_logs": ("Treatment log not found.", "treatment_log_not_found"),
    "notifications": ("Notification not found.", "notification_not_found"),
    "user_preferences": ("Settings not found.", "settings_not_found"),
}


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class QueryResult:
    data: list[dict[str, Any]]


class DatabaseClient:
    def __init__(self, connection: Any, *, dialect: str) -> None:
        self.connection = connection
        self.dialect = dialect
        self._in_transaction = False

    @classmethod
    def connect(cls, database_url: str | None = None) -> "DatabaseClient":
        database_url = database_url or settings.database_url
        if database_url.startswith("sqlite:///"):
            return cls(_connect_sqlite(database_url), dialect="sqlite")
        if database_url.startswith("mysql://") or database_url.startswith("mysql+pymysql://"):
            return cls(_connect_mysql(database_url), dialect="mysql")
        raise AppException(
            message="Unsupported database URL.",
            code="unsupported_database_url",
            details={"database_url": database_url},
            status_code=500,
        )

    @property
    def placeholder(self) -> str:
        return "?" if self.dialect == "sqlite" else "%s"

    def close(self) -> None:
        self.connection.close()

    def commit(self) -> None:
        self.connection.commit()

    def rollback(self) -> None:
        self.connection.rollback()

    @contextmanager
    def transaction(self):
        previous_state = self._in_transaction
        self._in_transaction = True
        try:
            yield self
            if not previous_state:
                self.commit()
        except Exception:
            if not previous_state:
                self.rollback()
            raise
        finally:
            self._in_transaction = previous_state

    def table(self, table_name: str) -> "QueryBuilder":
        return QueryBuilder(self, table_name)

    def _execute(self, sql: str, params: list[Any] | tuple[Any, ...], *, fetch: bool) -> list[dict[str, Any]]:
        cursor = self.connection.cursor()
        try:
            cursor.execute(sql, params)
            if fetch:
                if self.dialect == "sqlite":
                    rows = [dict(row) for row in cursor.fetchall()]
                else:
                    rows = list(cursor.fetchall())
            else:
                rows = []
            if not self._in_transaction and not fetch:
                self.commit()
            return [self._normalize_row(row) for row in rows]
        except Exception as exc:
            if not self._in_transaction:
                self.rollback()
            raise AppException(
                message="Database operation failed.",
                code="database_operation_failed",
                details={"reason": str(exc), "sql": sql},
                status_code=500,
            ) from exc
        finally:
            cursor.close()

    def _normalize_row(self, row: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, datetime):
                normalized[key] = value.replace(tzinfo=UTC).isoformat() if value.tzinfo is None else value.isoformat()
            elif isinstance(value, date):
                normalized[key] = value.isoformat()
            elif isinstance(value, Decimal):
                normalized[key] = float(value)
            elif isinstance(value, bytes):
                normalized[key] = value.decode("utf-8")
            elif key in BOOLEAN_COLUMNS and value is not None:
                normalized[key] = bool(value)
            else:
                normalized[key] = value

        for table_name, json_columns in JSON_COLUMNS.items():
            if not json_columns.intersection(normalized.keys()):
                continue
            for column in json_columns:
                value = normalized.get(column)
                if isinstance(value, str) and value:
                    try:
                        normalized[column] = json.loads(value)
                    except json.JSONDecodeError:
                        pass
        return normalized


class QueryBuilder:
    def __init__(self, client: DatabaseClient, table_name: str) -> None:
        self.client = client
        self.table_name = table_name
        self._action = "select"
        self._select_columns = "*"
        self._filters: list[tuple[str, str, Any]] = []
        self._limit: int | None = None
        self._order_by: tuple[str, bool] | None = None
        self._payload: dict[str, Any] | None = None

    def select(self, columns: str) -> "QueryBuilder":
        self._action = "select"
        self._select_columns = columns
        return self

    def insert(self, payload: dict[str, Any]) -> "QueryBuilder":
        self._action = "insert"
        self._payload = dict(payload)
        return self

    def update(self, payload: dict[str, Any]) -> "QueryBuilder":
        self._action = "update"
        self._payload = dict(payload)
        return self

    def delete(self) -> "QueryBuilder":
        self._action = "delete"
        return self

    def eq(self, column: str, value: Any) -> "QueryBuilder":
        self._filters.append(("=", column, value))
        return self

    def neq(self, column: str, value: Any) -> "QueryBuilder":
        self._filters.append(("!=", column, value))
        return self

    def limit(self, value: int) -> "QueryBuilder":
        self._limit = value
        return self

    def order(self, column: str, *, desc: bool = False) -> "QueryBuilder":
        self._order_by = (column, desc)
        return self

    def execute(self) -> QueryResult:
        if self._action == "select":
            return QueryResult(data=self._execute_select())
        if self._action == "insert":
            return QueryResult(data=self._execute_insert())
        if self._action == "update":
            return QueryResult(data=self._execute_update())
        if self._action == "delete":
            return QueryResult(data=self._execute_delete())
        raise AppException(message="Unsupported database action.", code="unsupported_database_action", status_code=500)

    def _execute_select(self) -> list[dict[str, Any]]:
        sql = f"SELECT {self._select_columns} FROM `{self.table_name}`"
        where_sql, params = self._build_filters()
        sql += where_sql
        if self._order_by:
            column, desc = self._order_by
            sql += f" ORDER BY `{column}` {'DESC' if desc else 'ASC'}"
        if self._limit is not None:
            sql += f" LIMIT {int(self._limit)}"
        return self.client._execute(sql, params, fetch=True)

    def _execute_insert(self) -> list[dict[str, Any]]:
        payload = self._prepare_payload(for_update=False)
        columns = list(payload.keys())
        placeholders = ", ".join([self.client.placeholder] * len(columns))
        columns_sql = ", ".join(f"`{column}`" for column in columns)
        sql = f"INSERT INTO `{self.table_name}` ({columns_sql}) VALUES ({placeholders})"
        params = [payload[column] for column in columns]
        self.client._execute(sql, params, fetch=False)
        primary_key = PRIMARY_KEY_COLUMNS.get(self.table_name, "id")
        return self.client.table(self.table_name).select("*").eq(primary_key, payload[primary_key]).limit(1).execute().data

    def _execute_update(self) -> list[dict[str, Any]]:
        payload = self._prepare_payload(for_update=True)
        if not payload:
            return self._execute_select()
        assignments = ", ".join(f"`{column}` = {self.client.placeholder}" for column in payload)
        params = [payload[column] for column in payload]
        where_sql, where_params = self._build_filters()
        sql = f"UPDATE `{self.table_name}` SET {assignments}{where_sql}"
        self.client._execute(sql, params + where_params, fetch=False)
        return self._execute_select()

    def _execute_delete(self) -> list[dict[str, Any]]:
        rows = self._execute_select()
        where_sql, params = self._build_filters()
        sql = f"DELETE FROM `{self.table_name}`{where_sql}"
        self.client._execute(sql, params, fetch=False)
        return rows

    def _prepare_payload(self, *, for_update: bool) -> dict[str, Any]:
        payload = dict(self._payload or {})
        primary_key = PRIMARY_KEY_COLUMNS.get(self.table_name, "id")
        if not for_update and primary_key == "id" and "id" not in payload:
            payload["id"] = str(uuid4())
        if self.table_name in UPDATED_AT_TABLES:
            payload["updated_at"] = _utc_now_iso()
        return {key: self._serialize_value(key, value) for key, value in payload.items()}

    def _serialize_value(self, column: str, value: Any) -> Any:
        if hasattr(value, "value"):
            value = value.value
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        if isinstance(value, bool):
            return int(value) if self.client.dialect == "sqlite" else value
        return value

    def _build_filters(self) -> tuple[str, list[Any]]:
        if not self._filters:
            return "", []
        clauses: list[str] = []
        params: list[Any] = []
        for operator, column, value in self._filters:
            clauses.append(f"`{column}` {operator} {self.client.placeholder}")
            params.append(self._serialize_value(column, value))
        return f" WHERE {' AND '.join(clauses)}", params


def _connect_sqlite(database_url: str):
    raw_path = database_url.removeprefix("sqlite:///")
    if raw_path != ":memory:":
        path = Path(raw_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        path.parent.mkdir(parents=True, exist_ok=True)
        raw_path = str(path)
    connection = sqlite3.connect(raw_path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def _connect_mysql(database_url: str):
    try:
        import pymysql
        from pymysql.cursors import DictCursor
    except ImportError as exc:
        raise AppException(
            message="PyMySQL is required for MySQL/XAMPP connections.",
            code="missing_mysql_driver",
            status_code=500,
        ) from exc

    normalized_url = database_url.replace("mysql+pymysql://", "mysql://", 1)
    parsed = urlparse(normalized_url)
    database_name = parsed.path.lstrip("/")
    return pymysql.connect(
        host=parsed.hostname or "127.0.0.1",
        user=unquote(parsed.username or "root"),
        password=unquote(parsed.password or ""),
        database=unquote(database_name),
        port=parsed.port or 3306,
        charset="utf8mb4",
        autocommit=False,
        cursorclass=DictCursor,
    )


def fetch_owned_record(client: DatabaseClient, table: str, record_id: str, user_id: str) -> dict[str, Any]:
    result = client.table(table).select("*").eq("id", record_id).eq("user_id", user_id).limit(1).execute()
    rows = result.data or []
    if not rows:
        message, code = RESOURCE_NOT_FOUND_META.get(
            table,
            (f"{table.rstrip('s').replace('_', ' ').title()} not found.", "record_not_found"),
        )
        raise NotFoundException(message=message, code=code)
    return rows[0]

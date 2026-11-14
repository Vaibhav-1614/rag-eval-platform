"""
Feedback rows and the shared daily LLM-call counter.

Two backends with the same interface:
  * SQLite (default): a local file. Fine for one container, but free hosts such as
    Hugging Face Spaces wipe the disk on restart.
  * Postgres: used when DATABASE_URL is set (any Postgres; the free tiers of Neon or
    Supabase work). Survives restarts and is shared by several containers.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class FeedbackStore:
    """SQLite backend."""

    backend = "sqlite"

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(path)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS feedback (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       request_id TEXT NOT NULL,
                       rating INTEGER NOT NULL,
                       comment TEXT,
                       created_at TEXT NOT NULL)"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS llm_usage (
                       day TEXT PRIMARY KEY,
                       calls INTEGER NOT NULL)"""
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, timeout=5)

    def add_feedback(self, request_id: str, rating: int, comment: str | None) -> int:
        with self._lock, self._conn() as c:
            cur = c.execute(
                "INSERT INTO feedback (request_id, rating, comment, created_at) VALUES (?,?,?,?)",
                (request_id, rating, comment, _now()),
            )
            return int(cur.lastrowid or 0)

    def take_llm_call(self, cap: int) -> bool:
        """Atomically count one LLM call for today (UTC). False once the cap is reached."""
        day = _today()
        with self._lock, self._conn() as c:
            row = c.execute("SELECT calls FROM llm_usage WHERE day=?", (day,)).fetchone()
            if (row[0] if row else 0) >= cap:
                return False
            c.execute(
                "INSERT INTO llm_usage(day, calls) VALUES(?, 1) "
                "ON CONFLICT(day) DO UPDATE SET calls = calls + 1",
                (day,),
            )
            return True

    def llm_calls_today(self) -> int:
        with self._conn() as c:
            row = c.execute("SELECT calls FROM llm_usage WHERE day=?", (_today(),)).fetchone()
            return row[0] if row else 0


class PostgresStore:
    """Postgres backend (psycopg 3). One short-lived connection per operation: the
    service makes a handful of writes per request, and serverless Postgres (Neon,
    Supabase) closes idle connections anyway."""

    backend = "postgres"

    def __init__(self, url: str) -> None:
        import psycopg

        self._psycopg = psycopg
        self._url = url
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS feedback (
                       id BIGSERIAL PRIMARY KEY,
                       request_id TEXT NOT NULL,
                       rating SMALLINT NOT NULL,
                       comment TEXT,
                       created_at TIMESTAMPTZ NOT NULL DEFAULT now())"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS llm_usage (
                       day DATE PRIMARY KEY,
                       calls INTEGER NOT NULL)"""
            )

    def _conn(self):
        return self._psycopg.connect(self._url, connect_timeout=5, autocommit=True)

    def add_feedback(self, request_id: str, rating: int, comment: str | None) -> int:
        with self._conn() as c:
            row = c.execute(
                "INSERT INTO feedback (request_id, rating, comment) VALUES (%s,%s,%s) RETURNING id",
                (request_id, rating, comment),
            ).fetchone()
            return int(row[0])

    def take_llm_call(self, cap: int) -> bool:
        if cap <= 0:
            return False
        # One statement, so concurrent containers cannot both take the last slot.
        with self._conn() as c:
            row = c.execute(
                """INSERT INTO llm_usage (day, calls) VALUES (%s, 1)
                   ON CONFLICT (day) DO UPDATE SET calls = llm_usage.calls + 1
                   WHERE llm_usage.calls < %s
                   RETURNING calls""",
                (_today(), cap),
            ).fetchone()
            return row is not None

    def llm_calls_today(self) -> int:
        with self._conn() as c:
            row = c.execute("SELECT calls FROM llm_usage WHERE day=%s", (_today(),)).fetchone()
            return int(row[0]) if row else 0


def make_store(database_url: str, sqlite_path: Path):
    return PostgresStore(database_url) if database_url else FeedbackStore(sqlite_path)

"""Small SQLite stores: user feedback and the shared daily LLM-call counter."""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


class FeedbackStore:
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
                (request_id, rating, comment, datetime.now(timezone.utc).isoformat()),
            )
            return int(cur.lastrowid or 0)

    def take_llm_call(self, cap: int) -> bool:
        """Atomically count one LLM call for today (UTC). False once the cap is reached."""
        day = datetime.now(timezone.utc).date().isoformat()
        with self._lock, self._conn() as c:
            row = c.execute("SELECT calls FROM llm_usage WHERE day=?", (day,)).fetchone()
            used = row[0] if row else 0
            if used >= cap:
                return False
            c.execute(
                "INSERT INTO llm_usage(day, calls) VALUES(?, 1) "
                "ON CONFLICT(day) DO UPDATE SET calls = calls + 1",
                (day,),
            )
            return True

    def llm_calls_today(self) -> int:
        day = datetime.now(timezone.utc).date().isoformat()
        with self._conn() as c:
            row = c.execute("SELECT calls FROM llm_usage WHERE day=?", (day,)).fetchone()
            return row[0] if row else 0

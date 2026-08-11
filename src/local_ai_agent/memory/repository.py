"""Durable long-term and semantic memory stored in SQLite with FTS5 retrieval."""

from __future__ import annotations

import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from local_ai_agent.db.schema import initialize_database
from local_ai_agent.schemas.contracts import ConfidenceLevel, MemoryRecord


class MemoryRepository:
    """Persist, retrieve, and age memory without treating unconfirmed content as fact."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path

    def initialize(self) -> None:
        with self._connect() as connection:
            initialize_database(connection)

    def upsert(self, memory: MemoryRecord) -> MemoryRecord:
        now = self._now()
        expires_at = memory.expires_at.isoformat() if memory.expires_at else None
        confidence = self._effective_confidence(memory.confidence, memory.expires_at)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memories (
                    category, memory_key, value, confidence, source_run_id, expires_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(category, memory_key) DO UPDATE SET
                    value = excluded.value,
                    confidence = excluded.confidence,
                    source_run_id = excluded.source_run_id,
                    expires_at = excluded.expires_at,
                    updated_at = excluded.updated_at
                """,
                (
                    memory.category.value,
                    memory.key,
                    memory.value,
                    confidence.value,
                    str(memory.source_run_id) if memory.source_run_id else None,
                    expires_at,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM memories WHERE category = ? AND memory_key = ?",
                (memory.category.value, memory.key),
            ).fetchone()
            self._refresh_fts(connection, int(row["id"]), memory.key, memory.value)
        return self._row_to_memory(row)

    def get(self, *, category: str, key: str) -> MemoryRecord | None:
        self.mark_expired_stale()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM memories WHERE category = ? AND memory_key = ?", (category, key)
            ).fetchone()
        return self._row_to_memory(row) if row else None

    def search(self, query: str, limit: int, include_stale: bool = False) -> list[MemoryRecord]:
        """Search memory values/keys through FTS5; query terms are sanitized into an OR expression."""
        self.mark_expired_stale()
        fts_query = self._fts_query(query)
        if not fts_query:
            return self.recent(limit=limit, include_stale=include_stale)
        stale_filter = "" if include_stale else "AND memories.confidence != 'STALE'"
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT memories.* FROM memory_fts
                JOIN memories ON memories.id = memory_fts.rowid
                WHERE memory_fts MATCH ? {stale_filter}
                ORDER BY bm25(memory_fts), memories.updated_at DESC
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def recent(self, limit: int, include_stale: bool = False) -> list[MemoryRecord]:
        self.mark_expired_stale()
        stale_filter = "" if include_stale else "WHERE confidence != 'STALE'"
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM memories {stale_filter} ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def mark_expired_stale(self) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE memories SET confidence = ?, updated_at = ?
                WHERE expires_at IS NOT NULL AND expires_at <= ? AND confidence != ?
                """,
                (
                    ConfidenceLevel.STALE.value,
                    self._now(),
                    self._now(),
                    ConfidenceLevel.STALE.value,
                ),
            )
        return int(cursor.rowcount)

    @staticmethod
    def _refresh_fts(connection: sqlite3.Connection, memory_id: int, key: str, value: str) -> None:
        connection.execute("DELETE FROM memory_fts WHERE rowid = ?", (memory_id,))
        connection.execute(
            "INSERT INTO memory_fts (rowid, memory_key, value) VALUES (?, ?, ?)",
            (memory_id, key, value),
        )

    @staticmethod
    def _fts_query(query: str) -> str:
        terms = re.findall(r"[\w-]{2,}", query.lower())
        return " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms)

    @staticmethod
    def _effective_confidence(
        confidence: ConfidenceLevel, expires_at: datetime | None
    ) -> ConfidenceLevel:
        if expires_at and expires_at <= datetime.now(UTC):
            return ConfidenceLevel.STALE
        return confidence

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            category=row["category"],
            key=row["memory_key"],
            value=row["value"],
            confidence=row["confidence"],
            source_run_id=row["source_run_id"],
            expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def _connect(self) -> sqlite3.Connection:
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._database_path, isolation_level="IMMEDIATE")
        connection.row_factory = sqlite3.Row
        return connection

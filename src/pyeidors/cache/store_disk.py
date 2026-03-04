"""Persistent disk cache store backed by sqlite index + object files."""

from __future__ import annotations

import gzip
import os
from pathlib import Path
import pickle
import sqlite3
import tempfile
import threading
import time
from typing import Any
from contextlib import contextmanager


class DiskCacheStore:
    """Persistent cache store with best-effort corruption recovery."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        max_bytes: int,
        compress_payloads: bool = True,
        read_only: bool = False,
        default_ttl_seconds: float | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.objects_dir = self.root_dir / "objects"
        self.db_path = self.root_dir / "index.sqlite"
        self.max_bytes = int(max(0, max_bytes))
        self.compress_payloads = bool(compress_payloads)
        self.read_only = bool(read_only)
        self.default_ttl_seconds = default_ttl_seconds
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    @contextmanager
    def _session(self):
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._session() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    artifact TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    cost REAL NOT NULL,
                    created_at REAL NOT NULL,
                    last_access REAL NOT NULL,
                    ttl_seconds REAL
                )
                """
            )
            conn.commit()

    def _entry_path(self, artifact: str, key: str) -> Path:
        art_dir = self.objects_dir / artifact
        art_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".pkl.gz" if self.compress_payloads else ".pkl"
        return art_dir / f"{key}{suffix}"

    def _serialize(self, value: Any) -> bytes:
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        if self.compress_payloads:
            return gzip.compress(payload, compresslevel=4)
        return payload

    def _deserialize(self, payload: bytes) -> Any:
        if self.compress_payloads:
            payload = gzip.decompress(payload)
        return pickle.loads(payload)

    def _is_expired(self, created_at: float, ttl_seconds: float | None) -> bool:
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        if ttl is None:
            return False
        return (time.time() - created_at) > ttl

    def get(self, key: str) -> Any | None:
        now = time.time()
        with self._lock:
            with self._session() as conn:
                row = conn.execute(
                    "SELECT artifact, file_path, created_at, ttl_seconds FROM cache_entries WHERE cache_key = ?",
                    (key,),
                ).fetchone()
                if row is None:
                    self.misses += 1
                    return None
                artifact, file_path, created_at, ttl_seconds = row
                file = Path(file_path)
                if self._is_expired(float(created_at), ttl_seconds):
                    self._remove_entry(conn, key, file)
                    conn.commit()
                    self.misses += 1
                    return None
                try:
                    payload = file.read_bytes()
                    value = self._deserialize(payload)
                except Exception:
                    self._remove_entry(conn, key, file)
                    conn.commit()
                    self.misses += 1
                    return None
                conn.execute(
                    "UPDATE cache_entries SET last_access = ? WHERE cache_key = ?",
                    (now, key),
                )
                conn.commit()
                self.hits += 1
                _ = artifact
                return value

    def put(
        self,
        key: str,
        value: Any,
        *,
        artifact: str,
        cost: float,
        ttl_seconds: float | None = None,
    ) -> bool:
        if self.read_only:
            return False
        try:
            payload = self._serialize(value)
        except Exception:
            return False

        target = self._entry_path(artifact, key)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=f".{key}.", dir=str(target.parent))
        os.close(fd)
        tmp = Path(tmp_path)
        tmp.write_bytes(payload)
        os.replace(tmp, target)
        size = int(target.stat().st_size)
        now = time.time()

        with self._lock:
            with self._session() as conn:
                conn.execute(
                    """
                    INSERT INTO cache_entries(cache_key, artifact, file_path, size_bytes, cost, created_at, last_access, ttl_seconds)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        artifact=excluded.artifact,
                        file_path=excluded.file_path,
                        size_bytes=excluded.size_bytes,
                        cost=excluded.cost,
                        created_at=excluded.created_at,
                        last_access=excluded.last_access,
                        ttl_seconds=excluded.ttl_seconds
                    """,
                    (key, artifact, str(target), size, float(cost), now, now, ttl_seconds),
                )
                conn.commit()
                self._evict_if_needed(conn)
                conn.commit()
        return True

    def _remove_entry(self, conn: sqlite3.Connection, key: str, file_path: Path) -> None:
        conn.execute("DELETE FROM cache_entries WHERE cache_key = ?", (key,))
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass

    def _evict_if_needed(self, conn: sqlite3.Connection) -> None:
        if self.max_bytes <= 0:
            for _, file_path in conn.execute("SELECT cache_key, file_path FROM cache_entries").fetchall():
                try:
                    Path(file_path).unlink(missing_ok=True)
                except Exception:
                    pass
            conn.execute("DELETE FROM cache_entries")
            return

        total_row = conn.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM cache_entries").fetchone()
        total = int(total_row[0] if total_row else 0)
        if total <= self.max_bytes:
            return
        rows = conn.execute(
            "SELECT cache_key, file_path, size_bytes FROM cache_entries ORDER BY cost ASC, last_access ASC"
        ).fetchall()
        for key, file_path, size in rows:
            self._remove_entry(conn, key, Path(file_path))
            total -= int(size)
            if total <= self.max_bytes:
                break

    def invalidate(self, prefix: str = "") -> int:
        with self._lock:
            with self._session() as conn:
                if prefix:
                    rows = conn.execute(
                        "SELECT cache_key, file_path FROM cache_entries WHERE cache_key LIKE ?",
                        (f"{prefix}%",),
                    ).fetchall()
                else:
                    rows = conn.execute("SELECT cache_key, file_path FROM cache_entries").fetchall()
                for key, file_path in rows:
                    self._remove_entry(conn, key, Path(file_path))
                conn.commit()
                return len(rows)

    def clear(self) -> None:
        self.invalidate(prefix="")

    def stats(self) -> dict[str, int]:
        with self._lock:
            with self._session() as conn:
                row = conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM cache_entries"
                ).fetchone()
                n_items = int(row[0] if row else 0)
                n_bytes = int(row[1] if row else 0)
        return {
            "hits": self.hits,
            "misses": self.misses,
            "items": n_items,
            "bytes": n_bytes,
            "max_bytes": int(self.max_bytes),
        }

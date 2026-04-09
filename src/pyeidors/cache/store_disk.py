"""Persistent disk cache store with EIDORS-style score-aware eviction."""

from __future__ import annotations

from contextlib import contextmanager
import gzip
import os
from pathlib import Path
import pickle
import sqlite3
import tempfile
import threading
import time
from typing import Any

from .types import compute_score_eff, compute_score_size


class DiskCacheStore:
    """Persistent cache store backed by sqlite + object files."""

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
                    name TEXT NOT NULL DEFAULT '',
                    namespace TEXT NOT NULL DEFAULT 'default',
                    file_path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    cost REAL NOT NULL,
                    effort REAL NOT NULL DEFAULT 1.0,
                    priority REAL NOT NULL DEFAULT 0.0,
                    use_count INTEGER NOT NULL DEFAULT 1,
                    score_eff REAL NOT NULL DEFAULT 0.0,
                    score_size REAL NOT NULL DEFAULT 0.0,
                    score REAL NOT NULL DEFAULT 0.0,
                    created_at REAL NOT NULL,
                    last_access REAL NOT NULL,
                    ttl_seconds REAL
                )
                """
            )
            self._ensure_schema_columns(conn)
            conn.commit()

    def _ensure_schema_columns(self, conn: sqlite3.Connection) -> None:
        existing = {
            row[1]: row for row in conn.execute("PRAGMA table_info(cache_entries)").fetchall()
        }
        migrations = (
            ("name", "ALTER TABLE cache_entries ADD COLUMN name TEXT NOT NULL DEFAULT ''"),
            (
                "namespace",
                "ALTER TABLE cache_entries ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'",
            ),
            ("effort", "ALTER TABLE cache_entries ADD COLUMN effort REAL NOT NULL DEFAULT 1.0"),
            ("priority", "ALTER TABLE cache_entries ADD COLUMN priority REAL NOT NULL DEFAULT 0.0"),
            ("use_count", "ALTER TABLE cache_entries ADD COLUMN use_count INTEGER NOT NULL DEFAULT 1"),
            ("score_eff", "ALTER TABLE cache_entries ADD COLUMN score_eff REAL NOT NULL DEFAULT 0.0"),
            ("score_size", "ALTER TABLE cache_entries ADD COLUMN score_size REAL NOT NULL DEFAULT 0.0"),
            ("score", "ALTER TABLE cache_entries ADD COLUMN score REAL NOT NULL DEFAULT 0.0"),
        )
        for column, statement in migrations:
            if column not in existing:
                conn.execute(statement)

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
                    """
                    SELECT file_path, created_at, ttl_seconds, effort, priority, use_count
                    FROM cache_entries WHERE cache_key = ?
                    """,
                    (key,),
                ).fetchone()
                if row is None:
                    self.misses += 1
                    return None

                file_path, created_at, ttl_seconds, effort, priority, use_count = row
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

                updated_use_count = int(use_count) + 1
                score_eff = compute_score_eff(
                    effort=float(effort),
                    use_count=updated_use_count,
                    priority=float(priority),
                )
                conn.execute(
                    """
                    UPDATE cache_entries
                    SET last_access = ?, use_count = ?, score_eff = ?, score = ?
                    WHERE cache_key = ?
                    """,
                    (now, updated_use_count, score_eff, score_eff, key),
                )
                conn.commit()
                self.hits += 1
                return value

    def put(
        self,
        key: str,
        value: Any,
        *,
        artifact: str,
        cost: float,
        ttl_seconds: float | None = None,
        name: str = "",
        namespace: str = "default",
        effort: float | None = None,
        priority: float = 0.0,
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
        use_effort = float(cost if effort is None else effort)
        score_eff = compute_score_eff(
            effort=use_effort,
            use_count=1,
            priority=float(priority),
        )
        score_size = compute_score_size(size)

        with self._lock:
            with self._session() as conn:
                conn.execute(
                    """
                    INSERT INTO cache_entries(
                        cache_key, artifact, name, namespace, file_path, size_bytes, cost,
                        effort, priority, use_count, score_eff, score_size, score,
                        created_at, last_access, ttl_seconds
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        artifact=excluded.artifact,
                        name=excluded.name,
                        namespace=excluded.namespace,
                        file_path=excluded.file_path,
                        size_bytes=excluded.size_bytes,
                        cost=excluded.cost,
                        effort=excluded.effort,
                        priority=excluded.priority,
                        use_count=excluded.use_count,
                        score_eff=excluded.score_eff,
                        score_size=excluded.score_size,
                        score=excluded.score,
                        created_at=excluded.created_at,
                        last_access=excluded.last_access,
                        ttl_seconds=excluded.ttl_seconds
                    """,
                    (
                        key,
                        artifact,
                        str(name),
                        str(namespace),
                        str(target),
                        size,
                        float(cost),
                        use_effort,
                        float(priority),
                        1,
                        score_eff,
                        score_size,
                        score_eff,
                        now,
                        now,
                        ttl_seconds,
                    ),
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
            rows = conn.execute("SELECT cache_key, file_path FROM cache_entries").fetchall()
            for key, file_path in rows:
                self._remove_entry(conn, key, Path(file_path))
            return

        total_row = conn.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM cache_entries").fetchone()
        total = int(total_row[0] if total_row else 0)
        if total <= self.max_bytes:
            return
        rows = conn.execute(
            """
            SELECT cache_key, file_path, size_bytes
            FROM cache_entries
            ORDER BY score_eff ASC, score_size DESC, last_access ASC
            """
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

    def clear_name(self, name: str, namespace: str | None = None) -> int:
        with self._lock:
            with self._session() as conn:
                if namespace is None:
                    rows = conn.execute(
                        "SELECT cache_key, file_path FROM cache_entries WHERE name = ?",
                        (name,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT cache_key, file_path
                        FROM cache_entries
                        WHERE name = ? AND namespace = ?
                        """,
                        (name, namespace),
                    ).fetchall()
                for key, file_path in rows:
                    self._remove_entry(conn, key, Path(file_path))
                conn.commit()
                return len(rows)

    def clear_max(self, max_bytes: int) -> int:
        target = int(max(0, max_bytes))
        with self._lock:
            with self._session() as conn:
                total_row = conn.execute(
                    "SELECT COALESCE(SUM(size_bytes), 0) FROM cache_entries"
                ).fetchone()
                total = int(total_row[0] if total_row else 0)
                if total <= target:
                    return 0
                rows = conn.execute(
                    """
                    SELECT cache_key, file_path, size_bytes
                    FROM cache_entries
                    ORDER BY score_eff ASC, score_size DESC, last_access ASC
                    """
                ).fetchall()
                removed = 0
                for key, file_path, size in rows:
                    self._remove_entry(conn, key, Path(file_path))
                    total -= int(size)
                    removed += 1
                    if total <= target:
                        break
                conn.commit()
                return removed

    def clear_old(self, timestamp: float) -> int:
        ts = float(timestamp)
        with self._lock:
            with self._session() as conn:
                rows = conn.execute(
                    """
                    SELECT cache_key, file_path
                    FROM cache_entries
                    WHERE last_access < ?
                    """,
                    (ts,),
                ).fetchall()
                for key, file_path in rows:
                    self._remove_entry(conn, key, Path(file_path))
                conn.commit()
                return len(rows)

    def clear_new(self, timestamp: float) -> int:
        ts = float(timestamp)
        with self._lock:
            with self._session() as conn:
                rows = conn.execute(
                    """
                    SELECT cache_key, file_path
                    FROM cache_entries
                    WHERE last_access > ?
                    """,
                    (ts,),
                ).fetchall()
                for key, file_path in rows:
                    self._remove_entry(conn, key, Path(file_path))
                conn.commit()
                return len(rows)

    def get_value(self, key: str) -> Any | None:
        with self._lock:
            with self._session() as conn:
                row = conn.execute(
                    "SELECT file_path FROM cache_entries WHERE cache_key = ?",
                    (key,),
                ).fetchone()
                if row is None:
                    return None
                file_path = Path(row[0])
        try:
            payload = file_path.read_bytes()
            return self._deserialize(payload)
        except Exception:
            return None

    def list_entries(
        self,
        *,
        name: str | None = None,
        namespace: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT
                cache_key, artifact, name, namespace, size_bytes, cost, effort,
                priority, use_count, score_eff, score_size, score, created_at, last_access
            FROM cache_entries
        """
        conditions: list[str] = []
        params: list[Any] = []
        if name is not None:
            conditions.append("name = ?")
            params.append(name)
        if namespace is not None:
            conditions.append("namespace = ?")
            params.append(namespace)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY last_access DESC"
        if limit is not None and limit > 0:
            query += " LIMIT ?"
            params.append(int(limit))

        with self._lock:
            with self._session() as conn:
                rows = conn.execute(query, tuple(params)).fetchall()
        return [
            {
                "key": row[0],
                "artifact": row[1],
                "name": row[2],
                "namespace": row[3],
                "size_bytes": int(row[4]),
                "cost": float(row[5]),
                "effort": float(row[6]),
                "priority": float(row[7]),
                "use_count": int(row[8]),
                "score_eff": float(row[9]),
                "score_size": float(row[10]),
                "score": float(row[11]),
                "created_at": float(row[12]),
                "last_access": float(row[13]),
                "layer": "disk",
            }
            for row in rows
        ]

    def collect_recent(
        self,
        *,
        names: list[str],
        limit_per_name: int = 1,
        namespace: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        collected: dict[str, list[dict[str, Any]]] = {}
        for name in names:
            collected[name] = self.list_entries(
                name=name,
                namespace=namespace,
                limit=max(1, int(limit_per_name)),
            )
        return collected

    def clear(self) -> None:
        self.invalidate(prefix="")

    def stats(self) -> dict[str, Any]:
        with self._lock:
            with self._session() as conn:
                row = conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM cache_entries"
                ).fetchone()
                n_items = int(row[0] if row else 0)
                n_bytes = int(row[1] if row else 0)
                by_artifact = {
                    str(item[0]): int(item[1])
                    for item in conn.execute(
                        "SELECT artifact, COUNT(*) FROM cache_entries GROUP BY artifact"
                    ).fetchall()
                }
                by_namespace = {
                    str(item[0]): int(item[1])
                    for item in conn.execute(
                        "SELECT namespace, COUNT(*) FROM cache_entries GROUP BY namespace"
                    ).fetchall()
                }
        return {
            "hits": self.hits,
            "misses": self.misses,
            "items": n_items,
            "bytes": n_bytes,
            "max_bytes": int(self.max_bytes),
            "artifacts": by_artifact,
            "namespaces": by_namespace,
        }

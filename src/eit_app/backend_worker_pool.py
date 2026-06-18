"""Persistent profile-scoped backend worker pool for the GUI."""

from __future__ import annotations

import atexit
from collections import deque
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid
from typing import Callable

from eit_app.backend_worker_runtime import (
    BackendWorkerCache,
    backend_worker_command,
    backend_worker_env,
    backend_worker_profile_root,
    clean_profile_command_env,
    looks_like_ffcx_jit_timeout,
    repair_ffcx_jit_timeout_cache,
)


class BackendWorkerTransportError(RuntimeError):
    """The persistent worker process or line protocol failed."""


class BackendWorkerRequestError(RuntimeError):
    """The worker executed the request and the solver returned an error."""


@dataclass(frozen=True)
class WorkerRunMetadata:
    profile: str
    cache_home: Path
    launch_mode: str
    pid: int
    reused_process: bool
    stale_jit_locks_removed: int
    rss_bytes: int = 0
    rss_limit_bytes: int = 0
    recycled_after_request: bool = False
    recycle_reason: str = ""
    primed_runtime: bool = False
    prime_command: str = ""
    prime_duration_ms: float = 0.0
    prime_metadata: dict[str, object] = field(default_factory=dict)
    request_duration_ms: float = 0.0


class _PersistentBackendWorker:
    def __init__(self, *, repo: Path, profile: str) -> None:
        self.repo = Path(repo)
        self.profile = str(profile or "default").strip() or "default"
        self._lock = threading.RLock()
        self._proc: subprocess.Popen[str] | None = None
        self._cache: BackendWorkerCache | None = None
        self._launch_mode = ""
        self._stderr_tail: deque[str] = deque(maxlen=50)
        self._stderr_thread: threading.Thread | None = None
        self._runtime_primed = False

    def run(
        self,
        *,
        command: str,
        input_path: Path,
        output_path: Path,
        progress_cb: Callable[[str], None] | None,
    ) -> WorkerRunMetadata:
        with self._lock:
            reused = self._is_running()
            if not reused:
                self._start(progress_cb=progress_cb)
            proc = self._proc
            cache = self._cache
            if proc is None or cache is None:
                raise BackendWorkerTransportError("backend worker did not start")
            request_id = uuid.uuid4().hex
            payload = {
                "id": request_id,
                "command": str(command),
                "input": str(input_path),
                "output": str(output_path),
            }
            request_started = time.perf_counter()
            message = self._send_payload(
                proc=proc,
                payload=payload,
                progress_cb=progress_cb,
            )
            request_duration_ms = (time.perf_counter() - request_started) * 1000.0
            if str(message.get("status", "")) == "ok":
                self._runtime_primed = True
                response_metadata = message.get("metadata", {})
                prime_metadata = (
                    dict(response_metadata)
                    if isinstance(response_metadata, dict)
                    else {}
                )
                prime_command = (
                    str(command)
                    if str(command) in {"prime_forward_setup", "prime_runtime"}
                    else ""
                )
                prime_duration_ms = request_duration_ms if prime_command else 0.0
                rss_bytes = _process_rss_bytes(int(proc.pid))
                rss_limit_bytes = _worker_max_rss_bytes()
                should_recycle = rss_limit_bytes > 0 and rss_bytes > rss_limit_bytes
                recycle_reason = "rss_budget_exceeded" if should_recycle else ""
                metadata = WorkerRunMetadata(
                    profile=self.profile,
                    cache_home=cache.xdg_cache_home,
                    launch_mode=self._launch_mode,
                    pid=int(proc.pid),
                    reused_process=reused,
                    stale_jit_locks_removed=len(cache.removed_stale_jit_locks),
                    rss_bytes=rss_bytes,
                    rss_limit_bytes=rss_limit_bytes,
                    recycled_after_request=should_recycle,
                    recycle_reason=recycle_reason,
                    primed_runtime=self._runtime_primed,
                    prime_command=prime_command,
                    prime_duration_ms=prime_duration_ms,
                    prime_metadata=prime_metadata,
                    request_duration_ms=request_duration_ms,
                )
                if should_recycle:
                    if progress_cb is not None:
                        progress_cb(
                            "Backend worker RSS exceeded budget; "
                            f"recycling profile={self.profile} "
                            f"rss={rss_bytes} limit={rss_limit_bytes}"
                        )
                    self.request_stop()
                return metadata
            error = str(message.get("error", "backend worker request failed"))
            raise BackendWorkerRequestError(error)

    def shutdown(self) -> None:
        with self._lock:
            self._stop()

    def warm(
        self, *, progress_cb: Callable[[str], None] | None = None
    ) -> WorkerRunMetadata:
        """Start the persistent worker process without running a solve."""

        with self._lock:
            reused = self._is_running()
            if not reused:
                self._start(progress_cb=progress_cb)
            proc = self._proc
            cache = self._cache
            if proc is None or cache is None:
                raise BackendWorkerTransportError("backend worker did not start")
            prime_command = "prime_runtime" if _worker_warm_prime_enabled() else ""
            prime_duration_ms = 0.0
            prime_metadata: dict[str, object] = {}
            if prime_command and not self._runtime_primed:
                started = time.perf_counter()
                request_id = uuid.uuid4().hex
                message = self._send_payload(
                    proc=proc,
                    payload={"id": request_id, "command": prime_command},
                    progress_cb=progress_cb,
                )
                prime_duration_ms = (time.perf_counter() - started) * 1000.0
                if str(message.get("status", "")) != "ok":
                    error = str(message.get("error", "backend runtime prime failed"))
                    raise BackendWorkerRequestError(error)
                metadata = message.get("metadata", {})
                if isinstance(metadata, dict):
                    prime_metadata = dict(metadata)
                self._runtime_primed = True
                if progress_cb is not None:
                    progress_cb(
                        "Primed backend worker runtime "
                        f"profile={self.profile} command={prime_command} "
                        f"in {prime_duration_ms:.0f} ms"
                    )
            rss_bytes = _process_rss_bytes(int(proc.pid))
            rss_limit_bytes = _worker_max_rss_bytes()
            should_recycle = rss_limit_bytes > 0 and rss_bytes > rss_limit_bytes
            recycle_reason = "rss_budget_exceeded" if should_recycle else ""
            metadata = WorkerRunMetadata(
                profile=self.profile,
                cache_home=cache.xdg_cache_home,
                launch_mode=self._launch_mode,
                pid=int(proc.pid),
                reused_process=reused,
                stale_jit_locks_removed=len(cache.removed_stale_jit_locks),
                rss_bytes=rss_bytes,
                rss_limit_bytes=rss_limit_bytes,
                recycled_after_request=should_recycle,
                recycle_reason=recycle_reason,
                primed_runtime=self._runtime_primed,
                prime_command=prime_command,
                prime_duration_ms=prime_duration_ms,
                prime_metadata=prime_metadata,
            )
            if should_recycle:
                if progress_cb is not None:
                    progress_cb(
                        "Backend worker warm RSS exceeded budget; "
                        f"recycling profile={self.profile} "
                        f"rss={rss_bytes} limit={rss_limit_bytes}"
                    )
                self.request_stop()
            return metadata

    def request_stop(self) -> None:
        """Stop the process without waiting for a concurrent request lock.

        GUI cancellation can happen while ``run`` is blocked in
        ``proc.stdout.readline()``. Taking ``_lock`` here would deadlock against
        that reader, so this path tears down the child process directly and lets
        the blocked request observe EOF.
        """
        proc = self._proc
        self._proc = None
        self._cache = None
        self._launch_mode = ""
        self._runtime_primed = False
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
        except OSError:
            return
        try:
            proc.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                return
            proc.wait(timeout=1.0)

    def _is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _start(self, *, progress_cb: Callable[[str], None] | None) -> None:
        env, cache = backend_worker_env(repo=self.repo, profile=self.profile)
        cmd, launch_mode = backend_worker_command(
            profile=self.profile,
            worker_args=["serve"],
        )
        if launch_mode == "profile_command":
            clean_profile_command_env(env)
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.repo),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise BackendWorkerTransportError(
                f"failed to start backend worker profile={self.profile}: {exc}"
            ) from exc
        self._proc = proc
        self._cache = cache
        self._launch_mode = launch_mode
        self._runtime_primed = False
        self._stderr_tail.clear()
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(proc,),
            name=f"eit-backend-worker-stderr-{self.profile}",
            daemon=True,
        )
        self._stderr_thread.start()
        if progress_cb is not None:
            progress_cb(
                "Started persistent backend worker "
                f"profile={self.profile} pid={proc.pid} via {launch_mode}"
            )
            progress_cb(f"Backend cache: {cache.xdg_cache_home}")
            if cache.removed_stale_jit_locks:
                progress_cb(
                    "Cleaned backend JIT cache: "
                    f"{len(cache.removed_stale_jit_locks)} stale lock file(s)."
                )

    def _drain_stderr(self, proc: subprocess.Popen[str]) -> None:
        stream = proc.stderr
        if stream is None:
            return
        try:
            for raw in stream:
                line = raw.rstrip()
                if line:
                    self._stderr_tail.append(line)
        except ValueError:
            return

    def _send_payload(
        self,
        *,
        proc: subprocess.Popen[str],
        payload: dict[str, object],
        progress_cb: Callable[[str], None] | None,
    ) -> dict[str, object]:
        request_id = str(payload.get("id", ""))
        if proc.stdin is None or proc.stdout is None:
            self._stop()
            raise BackendWorkerTransportError("backend worker pipes are closed")
        try:
            proc.stdin.write(json.dumps(payload, sort_keys=True) + "\n")
            proc.stdin.flush()
        except OSError as exc:
            self._stop()
            raise BackendWorkerTransportError(
                f"failed to write backend worker request: {exc}"
            ) from exc

        while True:
            try:
                line = proc.stdout.readline()
            except RuntimeError as exc:
                self.request_stop()
                raise BackendWorkerTransportError(
                    f"backend worker stdout read failed: {exc}"
                ) from exc
            if line == "":
                code = proc.poll()
                self._stop()
                tail = "\n".join(self._stderr_tail)
                raise BackendWorkerTransportError(
                    f"backend worker exited before replying (code={code}): {tail}"
                )
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                if progress_cb is not None:
                    progress_cb(line)
                continue
            if str(message.get("id", "")) != request_id:
                continue
            msg_type = str(message.get("type", ""))
            if msg_type == "progress":
                if progress_cb is not None:
                    progress_cb(str(message.get("message", "")))
                continue
            if msg_type == "done":
                return message

    def _stop(self) -> None:
        proc = self._proc
        self._proc = None
        self._cache = None
        self._launch_mode = ""
        self._runtime_primed = False
        if proc is None:
            return
        if proc.poll() is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.write(
                        json.dumps(
                            {
                                "id": uuid.uuid4().hex,
                                "command": "shutdown",
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    proc.stdin.flush()
            except OSError:
                pass
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)


_POOL_LOCK = threading.RLock()
_POOL: dict[tuple[str, str], _PersistentBackendWorker] = {}


def _profile_allows_persistent_worker(profile: str | None) -> bool:
    profile_name = str(profile or "").strip().lower()
    if "amgx" not in profile_name:
        return True
    raw = os.getenv("EIT_APP_BACKEND_WORKER_PERSISTENT_AMGX", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def persistent_backend_workers_enabled(profile: str | None = None) -> bool:
    raw = os.getenv("EIT_APP_BACKEND_WORKER_PERSISTENT", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"} and _profile_allows_persistent_worker(
        profile
    )


def _worker_warm_prime_enabled() -> bool:
    raw = os.getenv("EIT_APP_BACKEND_WORKER_WARM_PRIME", "1").strip().lower()
    return raw not in {"", "0", "false", "no", "off", "none", "disabled"}


def _worker_max_rss_bytes() -> int:
    raw = os.getenv("EIT_APP_BACKEND_WORKER_MAX_RSS_MB", "4096").strip().lower()
    if raw in {"", "0", "false", "no", "off", "none", "disabled"}:
        return 0
    try:
        mib = float(raw)
    except ValueError:
        mib = 4096.0
    if mib <= 0:
        return 0
    return int(mib * 1024 * 1024)


def _process_rss_bytes(pid: int) -> int:
    """Return current resident set size for a Linux process, or 0 if unknown."""

    try:
        with open(f"/proc/{int(pid)}/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except (OSError, ValueError):
        return 0
    return 0


def run_persistent_backend_worker_request(
    *,
    repo: Path,
    profile: str,
    command: str,
    input_path: Path,
    output_path: Path,
    progress_cb: Callable[[str], None] | None = None,
) -> WorkerRunMetadata:
    profile_name = str(profile or "default").strip() or "default"
    if not persistent_backend_workers_enabled(profile_name):
        raise BackendWorkerTransportError(
            f"persistent backend worker disabled for profile={profile_name}"
        )
    key = (str(Path(repo).resolve()), profile_name)
    repo_path = Path(repo)
    for attempt in range(2):
        with _POOL_LOCK:
            worker = _POOL.get(key)
            if worker is None:
                worker = _PersistentBackendWorker(repo=repo_path, profile=key[1])
                _POOL[key] = worker
        try:
            return worker.run(
                command=command,
                input_path=Path(input_path),
                output_path=Path(output_path),
                progress_cb=progress_cb,
            )
        except BackendWorkerTransportError:
            with _POOL_LOCK:
                if _POOL.get(key) is worker:
                    _POOL.pop(key, None)
            worker.shutdown()
            raise
        except BackendWorkerRequestError as exc:
            error = str(exc)
            if attempt == 0 and looks_like_ffcx_jit_timeout(error):
                with _POOL_LOCK:
                    if _POOL.get(key) is worker:
                        _POOL.pop(key, None)
                worker.request_stop()
                cache_home = (
                    backend_worker_profile_root(repo_path, key[1]) / "xdg-cache"
                )
                repaired = repair_ffcx_jit_timeout_cache(cache_home, error)
                if progress_cb is not None:
                    progress_cb(
                        "Recovered backend FFCx JIT cache after timeout "
                        f"(profile={key[1]}, repaired={len(repaired)}); retrying once."
                    )
                continue
            raise
    raise BackendWorkerTransportError("backend worker retry loop exhausted")


def stop_persistent_backend_worker(*, repo: Path, profile: str) -> bool:
    """Stop and evict a profile-scoped persistent worker if it exists."""

    key = (str(Path(repo).resolve()), str(profile or "default").strip() or "default")
    with _POOL_LOCK:
        worker = _POOL.pop(key, None)
    if worker is None:
        return False
    worker.request_stop()
    return True


def warm_persistent_backend_worker(
    *,
    repo: Path,
    profile: str,
    progress_cb: Callable[[str], None] | None = None,
) -> WorkerRunMetadata | None:
    """Warm a profile worker process without executing a forward/recon request."""

    profile_name = str(profile or "default").strip() or "default"
    if not persistent_backend_workers_enabled(profile_name):
        return None
    key = (str(Path(repo).resolve()), profile_name)
    with _POOL_LOCK:
        worker = _POOL.get(key)
        if worker is None:
            worker = _PersistentBackendWorker(repo=Path(repo), profile=key[1])
            _POOL[key] = worker
    try:
        return worker.warm(progress_cb=progress_cb)
    except BackendWorkerTransportError:
        with _POOL_LOCK:
            if _POOL.get(key) is worker:
                _POOL.pop(key, None)
        worker.shutdown()
        raise


def prime_persistent_backend_worker_forward_setup(
    *,
    repo: Path,
    profile: str,
    input_path: Path,
    progress_cb: Callable[[str], None] | None = None,
) -> WorkerRunMetadata | None:
    """Warm a worker by building compatible forward static setup only."""

    profile_name = str(profile or "default").strip() or "default"
    if not persistent_backend_workers_enabled(profile_name):
        return None
    return run_persistent_backend_worker_request(
        repo=Path(repo),
        profile=profile_name,
        command="prime_forward_setup",
        input_path=Path(input_path),
        output_path=Path(input_path).with_suffix(".prime.out"),
        progress_cb=progress_cb,
    )


def shutdown_persistent_backend_workers() -> None:
    with _POOL_LOCK:
        workers = list(_POOL.values())
        _POOL.clear()
    for worker in workers:
        worker.shutdown()


atexit.register(shutdown_persistent_backend_workers)

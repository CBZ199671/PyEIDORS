"""Unit tests for terminal-scoped cache shell helper."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "env" / "cache_session.sh"


def _run_bash(script: str, *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-lc", script],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_shell_helper_exports_session_env_and_reuses_it_within_one_shell(
    tmp_path: Path,
):
    cache_root = tmp_path / "cache-root"
    env = os.environ.copy()
    env["CACHE_ROOT"] = str(cache_root)
    env["META"] = str(tmp_path / "meta.txt")
    env["OUT1"] = str(tmp_path / "out1.txt")
    env["OUT2"] = str(tmp_path / "out2.txt")

    result = _run_bash(
        f'''
set -euo pipefail
source "{SCRIPT}"
deactivate() {{ :; }}
pyeidors_cache_session_init "$CACHE_ROOT"
printf '%s\n%s\n%s\n%s\n' \
  "$PYEIDORS_CACHE_SESSION_ID" \
  "$PYEIDORS_CACHE_SESSION_DIR" \
  "$PYEIDORS_CACHE_REQUESTED_ROOT" \
  "$PYEIDORS_CACHE_OWNER_PID" > "$META"
python - <<'PY2' > "$OUT1"
import os
print(os.environ["PYEIDORS_CACHE_SESSION_DIR"])
PY2
python - <<'PY2' > "$OUT2"
import os
print(os.environ["PYEIDORS_CACHE_SESSION_DIR"])
PY2
test "$(cat "$OUT1")" = "$(cat "$OUT2")"
''',
        env=env,
    )

    assert result.returncode == 0, result.stderr
    meta = (tmp_path / "meta.txt").read_text(encoding="utf-8").splitlines()
    assert meta[0].startswith("session-shellpid")
    assert Path(meta[1]).parent == cache_root.resolve() / ".sessions"
    assert not Path(meta[1]).exists()
    assert Path(meta[2]) == cache_root.resolve()
    assert meta[3].isdigit()
    assert (tmp_path / "out1.txt").read_text(encoding="utf-8").strip() == meta[1]
    assert (tmp_path / "out2.txt").read_text(encoding="utf-8").strip() == meta[1]


def test_shell_helper_cleans_session_on_deactivate_and_keeps_terminals_isolated(
    tmp_path: Path,
):
    cache_root = tmp_path / "cache-root"
    env = os.environ.copy()
    env["CACHE_ROOT"] = str(cache_root)
    env["META1"] = str(tmp_path / "meta1.txt")
    env["META2"] = str(tmp_path / "meta2.txt")

    result1 = _run_bash(
        f'''
set -euo pipefail
source "{SCRIPT}"
deactivate() {{ :; }}
pyeidors_cache_session_init "$CACHE_ROOT"
printf '%s\n' "$PYEIDORS_CACHE_SESSION_DIR" > "$META1"
deactivate
test ! -e "$(cat "$META1")"
''',
        env=env,
    )
    assert result1.returncode == 0, result1.stderr

    result2 = _run_bash(
        f'''
set -euo pipefail
source "{SCRIPT}"
deactivate() {{ :; }}
pyeidors_cache_session_init "$CACHE_ROOT"
printf '%s\n' "$PYEIDORS_CACHE_SESSION_DIR" > "$META2"
test -d "$(cat "$META2")"
''',
        env=env,
    )
    assert result2.returncode == 0, result2.stderr

    session1 = Path((tmp_path / "meta1.txt").read_text(encoding="utf-8").strip())
    session2 = Path((tmp_path / "meta2.txt").read_text(encoding="utf-8").strip())
    assert session1 != session2
    assert not session1.exists()
    assert not session2.exists()


def test_shell_helper_cleans_stale_dead_session_dirs_on_init(tmp_path: Path):
    cache_root = tmp_path / "cache-root"
    stale_dir = cache_root / ".sessions" / "session-shellpid999999-dead"
    stale_dir.mkdir(parents=True)
    (stale_dir / "payload.bin").write_text("x", encoding="utf-8")

    env = os.environ.copy()
    env["CACHE_ROOT"] = str(cache_root)

    result = _run_bash(
        f'''
set -euo pipefail
source "{SCRIPT}"
deactivate() {{ :; }}
pyeidors_cache_session_init "$CACHE_ROOT"
test ! -e "{stale_dir}"
''',
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert not stale_dir.exists()

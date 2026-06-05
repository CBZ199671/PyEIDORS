#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

VERSION="${1:-}"
if [ -z "$VERSION" ]; then
  VERSION="$(
    python3 - <<'PY'
from pathlib import Path
import tomllib

project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
print(project["version"])
PY
  )"
fi

DIST_DIR="$ROOT_DIR/dist"
STAGE_PARENT="$DIST_DIR/private-nix-source"
STAGE_DIR="$STAGE_PARENT/PyEIDORS-$VERSION"
ZIP_PATH="$DIST_DIR/PyEIDORS-$VERSION-pure-nix-source.zip"
SHA_FILE="$DIST_DIR/PyEIDORS-$VERSION-pure-nix-source.SHA256SUMS.txt"

RUN_TESTS="${RUN_TESTS:-1}"
RUN_WARM="${RUN_WARM:-1}"
RUN_CUDA_BUILD="${RUN_CUDA_BUILD:-0}"
CUDA_BUILD_LOG="${CUDA_BUILD_LOG:-$DIST_DIR/cuda-build.log}"
CUDA_PROGRESS_LOG="${CUDA_PROGRESS_LOG:-$DIST_DIR/cuda-build-progress.log}"
NIX_FLAGS=(--extra-experimental-features "nix-command flakes")
CPU_PACKAGE_ATTRS=(pyeidors pyeidors-complex pyeidors-complex64)
CPU_CACHE_APPS=(eit-cache eit-cache-complex eit-cache-complex64)
CPU_PROFILES=(default complex complex64)
CPU_SCALARS=(float64 complex128 complex64)
CUDA_PACKAGE_ATTRS=(pyeidors-cuda pyeidors-complex-cuda pyeidors-complex64-cuda)

record_cuda_progress() {
  if [ -x "$ROOT_DIR/scripts/release/monitor_cuda_build.sh" ]; then
    TAIL_LINES="${CUDA_MONITOR_TAIL_LINES:-120}" \
      "$ROOT_DIR/scripts/release/monitor_cuda_build.sh" \
      "$CUDA_BUILD_LOG" "$CUDA_PROGRESS_LOG" >/dev/null || true
  fi
}

validate_cpu_cache_apps() {
  local flake_ref="$1"
  local label="$2"
  local index app

  echo "[release] validating CPU cache apps from $label"
  for app in "${CPU_CACHE_APPS[@]}"; do
    nix "${NIX_FLAGS[@]}" run --option warn-dirty false "$flake_ref#$app" -- --help >/dev/null
  done

  if [ "$RUN_WARM" != "1" ]; then
    echo "[release] RUN_WARM=0, skipping backend worker warm validation for $label"
    return
  fi

  for index in "${!CPU_CACHE_APPS[@]}"; do
    app="${CPU_CACHE_APPS[$index]}"
    profile="${CPU_PROFILES[$index]}"
    expected_scalar="${CPU_SCALARS[$index]}"
    echo "[release] warming $app profile=$profile from $label"
    tmp="$(mktemp -d)"
    HOME="$tmp" XDG_CACHE_HOME="$tmp/cache" MPLCONFIGDIR="$tmp/mpl" \
      EIT_APP_GUI_RUNTIME_PROFILE="$profile" EIT_APP_BACKEND_WORKER_LAUNCH_MODE=direct \
      timeout 120s nix "${NIX_FLAGS[@]}" run --option warn-dirty false "$flake_ref#$app" -- \
      warm --profile "$profile" --cache-dir "$tmp/pyeidors-cache" > "$tmp/warm.json"
    python3 - "$tmp/warm.json" "$profile" "$expected_scalar" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
expected_profile = sys.argv[2]
expected_scalar = sys.argv[3]
data = json.loads(path.read_text(encoding="utf-8"))
actual_profile = data.get("profile")
scalar = data.get("prime_metadata", {}).get("scalar", {})
actual_scalar = scalar.get("petsc_scalar_type")
errors = data.get("prime_metadata", {}).get("errors", {})

if actual_profile != expected_profile:
    raise SystemExit(
        f"[release] ERROR: expected profile {expected_profile}, got {actual_profile}"
    )
if actual_scalar != expected_scalar:
    raise SystemExit(
        f"[release] ERROR: expected PETSc scalar {expected_scalar}, got {actual_scalar}"
    )
if errors:
    raise SystemExit(f"[release] ERROR: warm metadata contains errors: {errors}")
PY
    rm -rf "$tmp"
  done
}

validate_cuda_derivations() {
  local flake_ref="$1"
  local label="$2"
  local current_system attr

  current_system="$(nix "${NIX_FLAGS[@]}" eval --impure --raw --expr builtins.currentSystem)"
  if [ "$current_system" != "x86_64-linux" ]; then
    echo "[release] skipping CUDA derivation validation for $label on $current_system"
    return
  fi

  echo "[release] validating CUDA package derivations from $label"
  for attr in "${CUDA_PACKAGE_ATTRS[@]}"; do
    nix "${NIX_FLAGS[@]}" eval --raw "$flake_ref#packages.x86_64-linux.$attr.version" --option warn-dirty false >/dev/null
    nix "${NIX_FLAGS[@]}" path-info --derivation "$flake_ref#$attr" --option warn-dirty false >/dev/null
  done

  if [ "$RUN_CUDA_BUILD" = "1" ]; then
    echo "[release] RUN_CUDA_BUILD=1, building CUDA packages from $label"
    cuda_refs=()
    for attr in "${CUDA_PACKAGE_ATTRS[@]}"; do
      cuda_refs+=("$flake_ref#$attr")
    done
    mkdir -p "$(dirname "$CUDA_BUILD_LOG")" "$(dirname "$CUDA_PROGRESS_LOG")"
    {
      echo "===== CUDA build started: $(date -Is) ====="
      echo "label=$label"
      printf "refs:"
      printf " %s" "${cuda_refs[@]}"
      echo
    } | tee -a "$CUDA_BUILD_LOG"
    record_cuda_progress

    set +e
    nix "${NIX_FLAGS[@]}" build "${cuda_refs[@]}" \
      --option warn-dirty false --no-link --print-build-logs 2>&1 \
      | tee -a "$CUDA_BUILD_LOG"
    cuda_status="${PIPESTATUS[0]}"
    set -e

    {
      if [ "$cuda_status" -eq 0 ]; then
        echo "===== CUDA build finished successfully: $(date -Is) ====="
      else
        echo "===== CUDA build failed with status $cuda_status: $(date -Is) ====="
      fi
    } | tee -a "$CUDA_BUILD_LOG"
    record_cuda_progress
    return "$cuda_status"
  else
    echo "[release] RUN_CUDA_BUILD=0, CUDA packages were evaluated but not built"
  fi
}

echo "[release] preparing PyEIDORS $VERSION pure Nix source package"

if [ "$RUN_TESTS" = "1" ]; then
  echo "[release] running format/lint/regression checks"
  nix "${NIX_FLAGS[@]}" develop --command bash -lc \
    "uv run ruff format --check src/eit_app/backend_worker_runtime.py tests/unit/test_gui_backend_worker_routing.py && \
     uv run ruff check src/eit_app/backend_worker_runtime.py tests/unit/test_gui_backend_worker_routing.py && \
     uv run pytest tests/unit/test_gui_backend_worker_routing.py::test_v591_backend_worker_env_propagates_installed_site_packages -q --no-cov"
else
  echo "[release] RUN_TESTS=0, skipping uv-based checks"
fi

echo "[release] checking flake metadata"
nix "${NIX_FLAGS[@]}" flake check --option warn-dirty false --no-build

echo "[release] building package from working tree"
working_refs=()
for attr in "${CPU_PACKAGE_ATTRS[@]}"; do
  working_refs+=(".#$attr")
done
nix "${NIX_FLAGS[@]}" build "${working_refs[@]}" --option warn-dirty false --no-link --print-build-logs
validate_cpu_cache_apps "." "working tree"
validate_cuda_derivations "." "working tree"

echo "[release] staging minimal pure Nix source tree"
rm -rf "$STAGE_PARENT"
mkdir -p "$STAGE_DIR"

python3 - "$ROOT_DIR" "$STAGE_DIR" "$VERSION" <<'PY'
from __future__ import annotations

from pathlib import Path
import shutil
import sys

root = Path(sys.argv[1]).resolve()
stage = Path(sys.argv[2]).resolve()
version = sys.argv[3]

required_root_files = (
    "flake.nix",
    "flake.lock",
    "pyproject.toml",
    "README.md",
    "LICENSE",
)

for relative in required_root_files:
    source = root / relative
    if not source.exists():
        raise SystemExit(f"[release] ERROR: missing required file: {relative}")
    shutil.copy2(source, stage / relative)

docs_dir = stage / "docs"
docs_dir.mkdir(parents=True, exist_ok=True)
install_doc = root / "docs" / "PRIVATE_RUNTIME_INSTALL.md"
if install_doc.exists():
    shutil.copy2(install_doc, stage / "INSTALL.zh.md")
    shutil.copy2(install_doc, docs_dir / "PRIVATE_RUNTIME_INSTALL.md")

def ignore_src(path: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        item = Path(path) / name
        if name == "__pycache__":
            ignored.add(name)
        elif name.endswith((".pyc", ".pyo")):
            ignored.add(name)
        elif item.is_dir() and name.endswith(".egg-info"):
            ignored.add(name)
        elif item.parent.name == "src" and name.startswith("hello."):
            ignored.add(name)
    return ignored

shutil.copytree(root / "src", stage / "src", ignore=ignore_src)

required_entries = (
    "src/pyeidors/__init__.py",
    "src/eit_app/__init__.py",
    "src/eit_app/app.py",
    "src/eit_app/assets/logo.svg",
)
missing = [entry for entry in required_entries if not (stage / entry).exists()]
if missing:
    print("[release] ERROR: staged package missing required entries:", file=sys.stderr)
    for entry in missing:
        print(f"  {entry}", file=sys.stderr)
    raise SystemExit(1)

for forbidden in (
    "data",
    "results",
    "outputs",
    "eit_meshes",
    ".pyeidors_cache",
    "tests",
    "notes",
    "reports",
    "archived",
    "compare_with_Eidors",
    "SoftwareX-PyEidors-Paper",
    "temp_abs_result",
    "Software_patent",
    "dist",
    ".venv",
):
    if (stage / forbidden).exists():
        raise SystemExit(f"[release] ERROR: forbidden staged path exists: {forbidden}")

version_text = (stage / "pyproject.toml").read_text(encoding="utf-8")
if f'version = "{version}"' not in version_text:
    raise SystemExit("[release] ERROR: staged pyproject.toml version mismatch")

print("[release] staged pure Nix source tree validation passed")
PY

echo "[release] validating package from staged source tree"
nix "${NIX_FLAGS[@]}" flake check "path:$STAGE_DIR" --option warn-dirty false --no-build
staged_refs=()
for attr in "${CPU_PACKAGE_ATTRS[@]}"; do
  staged_refs+=("path:$STAGE_DIR#$attr")
done
nix "${NIX_FLAGS[@]}" build "${staged_refs[@]}" --option warn-dirty false --no-link --print-build-logs
validate_cpu_cache_apps "path:$STAGE_DIR" "staged source tree"
validate_cuda_derivations "path:$STAGE_DIR" "staged source tree"

echo "[release] writing $ZIP_PATH"
rm -f "$ZIP_PATH" "$SHA_FILE"
python3 - "$STAGE_PARENT" "PyEIDORS-$VERSION" "$ZIP_PATH" <<'PY'
from __future__ import annotations

from pathlib import Path
import sys
import zipfile

stage_parent = Path(sys.argv[1]).resolve()
top_name = sys.argv[2]
zip_path = Path(sys.argv[3]).resolve()
root = stage_parent / top_name

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        archive.write(path, arcname=str(path.relative_to(stage_parent)))
PY

(
  cd "$DIST_DIR"
  sha256sum "$(basename "$ZIP_PATH")" > "$(basename "$SHA_FILE")"
)

echo "[release] wrote $ZIP_PATH"
echo "[release] wrote $SHA_FILE"
echo "[release] contents:"
python3 -m zipfile -l "$ZIP_PATH" | sed -n '1,120p'

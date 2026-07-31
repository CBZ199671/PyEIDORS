#!/usr/bin/env bash

# Shared by the self-extracting installer and the installed launcher. Keep this
# file compatible with Bash 4+ because it runs before the Nix runtime is active.

PYEIDORS_MIN_NIX_VERSION="${PYEIDORS_MIN_NIX_VERSION:-2.4}"

pyeidors_safe_system_path() {
  if [ -z "${PYEIDORS_ORIGINAL_PATH+x}" ]; then
    PYEIDORS_ORIGINAL_PATH="${PATH:-}"
    export PYEIDORS_ORIGINAL_PATH
  fi
  PATH="/usr/sbin:/usr/bin:/sbin:/bin"
  export PATH
}

pyeidors_tool_works() {
  local tool="$1"
  local candidate="$2"

  [ -x "$candidate" ] || return 1
  case "$tool" in
    tar) "$candidate" --version >/dev/null 2>&1 ;;
    zstd) "$candidate" --version >/dev/null 2>&1 ;;
    unzip) "$candidate" -v >/dev/null 2>&1 ;;
    curl) "$candidate" --version >/dev/null 2>&1 ;;
    sha256sum)
      printf 'pyeidors-tool-probe\n' | "$candidate" >/dev/null 2>&1
      ;;
    *) "$candidate" --version >/dev/null 2>&1 ;;
  esac
}

pyeidors_resolve_host_tool() {
  local tool="$1"
  local directory candidate

  for directory in /usr/bin /bin /usr/sbin /sbin; do
    candidate="$directory/$tool"
    if pyeidors_tool_works "$tool" "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  if [ -n "${PYEIDORS_ORIGINAL_PATH:-}" ]; then
    candidate="$(
      PATH="$PYEIDORS_ORIGINAL_PATH" command -v "$tool" 2>/dev/null || true
    )"
  else
    candidate="$(command -v "$tool" 2>/dev/null || true)"
  fi
  if [ -n "$candidate" ] && pyeidors_tool_works "$tool" "$candidate"; then
    printf '%s\n' "$candidate"
    return 0
  fi
  return 1
}

pyeidors_probe_archive_tools() {
  local tar_bin="$1"
  local zstd_bin="$2"
  local mktemp_bin rm_bin probe_dir listing status

  mktemp_bin="$(pyeidors_resolve_host_tool mktemp)" || return 1
  rm_bin="$(pyeidors_resolve_host_tool rm)" || return 1
  probe_dir="$("$mktemp_bin" -d "${TMPDIR:-/tmp}/pyeidors-tool-probe.XXXXXX")" \
    || return 1

  status=0
  printf 'PyEIDORS archive tool probe\n' > "$probe_dir/input.txt"
  "$tar_bin" -C "$probe_dir" -cf "$probe_dir/input.tar" input.txt \
    >/dev/null 2>&1 || status=1
  if [ "$status" -eq 0 ]; then
    "$zstd_bin" -q -f "$probe_dir/input.tar" -o "$probe_dir/input.tar.zst" \
      >/dev/null 2>&1 || status=1
  fi
  if [ "$status" -eq 0 ]; then
    "$zstd_bin" -q -t "$probe_dir/input.tar.zst" >/dev/null 2>&1 || status=1
  fi
  if [ "$status" -eq 0 ]; then
    listing="$(
      "$zstd_bin" -q -d -c "$probe_dir/input.tar.zst" 2>/dev/null \
        | "$tar_bin" -tf - 2>/dev/null
    )" || status=1
    [ "$listing" = "input.txt" ] || status=1
  fi
  "$rm_bin" -rf -- "$probe_dir"
  [ "$status" -eq 0 ]
}

pyeidors_nix_version_at_least() {
  local actual="$1"
  local minimum="$2"
  local actual_major actual_minor minimum_major minimum_minor

  actual_major="${actual%%.*}"
  actual_minor="${actual#*.}"
  actual_minor="${actual_minor%%.*}"
  minimum_major="${minimum%%.*}"
  minimum_minor="${minimum#*.}"
  minimum_minor="${minimum_minor%%.*}"

  [[ "$actual_major" =~ ^[0-9]+$ ]] || return 1
  [[ "$actual_minor" =~ ^[0-9]+$ ]] || return 1
  [[ "$minimum_major" =~ ^[0-9]+$ ]] || return 1
  [[ "$minimum_minor" =~ ^[0-9]+$ ]] || return 1

  if ((10#$actual_major > 10#$minimum_major)); then
    return 0
  fi
  if ((10#$actual_major < 10#$minimum_major)); then
    return 1
  fi
  ((10#$actual_minor >= 10#$minimum_minor))
}

pyeidors_validate_nix_candidate() {
  local nix_bin="$1"
  local nix_store_bin version_output version

  [ -n "$nix_bin" ] && [ "${nix_bin#/}" != "$nix_bin" ] || return 1
  [ -x "$nix_bin" ] || return 1
  nix_store_bin="${nix_bin%/*}/nix-store"
  [ -x "$nix_store_bin" ] || return 1

  version_output="$("$nix_bin" --version 2>/dev/null)" || return 1
  if [[ "$version_output" =~ ([0-9]+)\.([0-9]+)(\.[0-9]+)? ]]; then
    version="${BASH_REMATCH[0]}"
  else
    return 1
  fi
  pyeidors_nix_version_at_least "$version" "$PYEIDORS_MIN_NIX_VERSION" \
    || return 1
  "$nix_store_bin" --version >/dev/null 2>&1 || return 1
}

pyeidors_find_nix() {
  local candidate
  local path_candidate=""
  local candidates=()

  [ -n "${PYEIDORS_NIX_BIN:-}" ] && candidates+=("$PYEIDORS_NIX_BIN")
  candidates+=(
    "/nix/var/nix/profiles/default/bin/nix"
    "$HOME/.nix-profile/bin/nix"
  )
  if [ -n "${PYEIDORS_ORIGINAL_PATH:-}" ]; then
    path_candidate="$(
      PATH="$PYEIDORS_ORIGINAL_PATH" command -v nix 2>/dev/null || true
    )"
  else
    path_candidate="$(command -v nix 2>/dev/null || true)"
  fi
  [ -n "$path_candidate" ] && candidates+=("$path_candidate")

  for candidate in "${candidates[@]}"; do
    if pyeidors_validate_nix_candidate "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

pyeidors_clean_runtime_environment() {
  unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE PYTHONWARNINGS
  unset VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL
  unset _CE_CONDA _CE_M
  unset CUDA_HOME CUDA_PATH CUDA_ROOT CUDACXX
  unset PETSC_DIR SLEPC_DIR CMAKE_PREFIX_PATH
  unset LD_PRELOAD
  unset QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH QML2_IMPORT_PATH
  export PYTHONNOUSERSITE=1
}

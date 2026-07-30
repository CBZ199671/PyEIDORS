#!/usr/bin/env bash
# 进入真实 float64 Nix 环境并启动 Jupyter Kernel。
# Enter the real-float64 Nix profile and launch the Jupyter kernel.
set -euo pipefail

package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${package_dir}/../.." && pwd)"

if ! command -v nix >/dev/null 2>&1; then
  if [[ -r "${HOME}/.nix-profile/etc/profile.d/nix.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/.nix-profile/etc/profile.d/nix.sh"
  elif [[ -r /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh ]]; then
    # shellcheck disable=SC1091
    source /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh
  fi
fi

cd "${repository_root}"
exec nix develop .#default --command python -m ipykernel_launcher "$@"

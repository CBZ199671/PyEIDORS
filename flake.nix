{
  description = "PyEIDORS development shell with FEniCSx (DOLFINx) via Nix + uv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      lib = nixpkgs.lib;
      systems = [
        "aarch64-darwin"
        "x86_64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];
      forAllSystems = lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python313;
          py = python.pkgs;
          fenicsDolfinx = py."fenics-dolfinx".overridePythonAttrs (
            old: {
              nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ py.cmake ];
              doCheck = false;
              doInstallCheck = false;
            }
          );
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.uv
              python
              pkgs.openmpi
              pkgs.hdf5
              pkgs.gmsh
              pkgs.pkg-config
              pkgs.cmake
              pkgs.ninja
              pkgs.gfortran
              pkgs.openblas

              fenicsDolfinx
              py."fenics-basix"
              py."fenics-ffcx"
              py."fenics-ufl"
              py.mpi4py

              py.numpy
              py.scipy
              py.matplotlib
              py.pandas
              py.h5py
              py.pyyaml
              py.meshio
              py.gmsh
              py.pytest
              py."pytest-cov"
              py.black
              py.flake8
              pkgs.pre-commit
            ];

            shellHook = ''
              export UV_PYTHON="${python}/bin/python3"
              export UV_PYTHON_PREFERENCE=only-system
              export PYTHONNOUSERSITE=1
              export HDF5_DIR="${pkgs.hdf5}"

              if [ "$(uname -s)" = "Darwin" ]; then
                mapfile -t _darwin_linker_fix < <("$UV_PYTHON" - <<'PY'
import pathlib
import shlex
import sysconfig


def sanitize_flags(raw: str) -> tuple[str, int]:
    tokens = shlex.split(raw)
    keep: list[str] = []
    removed = 0
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "-L" and i + 1 < len(tokens):
            candidate = tokens[i + 1]
            if pathlib.Path(candidate).is_dir():
                keep.extend([token, candidate])
            else:
                removed += 1
            i += 2
            continue
        if token.startswith("-L") and len(token) > 2:
            candidate = token[2:]
            if pathlib.Path(candidate).is_dir():
                keep.append(token)
            else:
                removed += 1
            i += 1
            continue
        keep.append(token)
        i += 1
    return shlex.join(keep), removed


ldflags_clean, ldflags_removed = sanitize_flags(sysconfig.get_config_var("LDFLAGS") or "")
ldshared_clean, ldshared_removed = sanitize_flags(sysconfig.get_config_var("LDSHARED") or "")

print(ldflags_clean)
print(ldshared_clean)
print(ldflags_removed + ldshared_removed)
PY
                )

                export LDFLAGS="''${_darwin_linker_fix[0]}"
                export LDSHARED="''${_darwin_linker_fix[1]}"
                if [ "''${_darwin_linker_fix[2]}" -gt 0 ]; then
                  echo "[nix+uv] Darwin linker flags sanitized: removed ''${_darwin_linker_fix[2]} invalid -L entries."
                fi
              fi

              nix_python_mm="$("$UV_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

              recreate_venv=0
              if [ ! -d .venv ]; then
                recreate_venv=1
              elif [ -x .venv/bin/python ]; then
                venv_python_mm="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
                if [ -z "$venv_python_mm" ] || [ "$venv_python_mm" != "$nix_python_mm" ]; then
                  echo "[nix+uv] Rebuilding .venv because Python version changed (.venv=$venv_python_mm, nix=$nix_python_mm)."
                  recreate_venv=1
                fi
              else
                recreate_venv=1
              fi

              if [ "$recreate_venv" -eq 1 ]; then
                rm -rf .venv
                echo "[nix+uv] Creating .venv with access to Nix site-packages..."
                uv venv --python "$UV_PYTHON" --system-site-packages
              fi

              source .venv/bin/activate

              venv_site="$(".venv/bin/python" - <<'PY'
import site
paths = site.getsitepackages()
print(paths[0] if paths else "")
PY
)"
              if [ -n "$venv_site" ]; then
                case ":''${PYTHONPATH:-}:" in
                  *":$venv_site:"*) ;;
                  *)
                    export PYTHONPATH="$venv_site''${PYTHONPATH:+:$PYTHONPATH}"
                    ;;
                esac
              fi

              if [ -x scripts/env/sync_locked_env.sh ]; then
                echo "[nix+uv] Checking locked Python environment profile (torch+cuqi+dev)..."
                if ! scripts/env/sync_locked_env.sh --check; then
                  echo "[nix+uv] Drift detected. Attempting automatic repair..."
                  if ! scripts/env/sync_locked_env.sh --repair; then
                    echo "[nix+uv] ERROR: environment repair failed."
                    echo "[nix+uv] Manual repair command: scripts/env/sync_locked_env.sh --repair"
                    exit 1
                  fi
                fi
              else
                echo "[nix+uv] WARNING: scripts/env/sync_locked_env.sh not found; skipping env sync."
              fi

              echo "[nix+uv] Dev shell ready."
              echo "[nix+uv] Verify stack quickly:"
              echo "  python -c \"import dolfinx, torch, cuqi, pyeidors; print(dolfinx.__version__)\""
            '';
          };
        }
      );
    };
}

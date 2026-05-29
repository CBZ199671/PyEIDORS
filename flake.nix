{
  description = "PyEIDORS development shells with FEniCSx (DOLFINx) via Nix + uv";

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
          nixpkgsPath = nixpkgs.outPath;
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python313;
          py = python.pkgs;
          linuxGuiLibs = [
            pkgs.glib
            pkgs.dbus
            pkgs.wayland
            pkgs.fontconfig
            pkgs.freetype
            pkgs.expat
            pkgs.xorg.libX11
            pkgs.xorg.libXau
            pkgs.xorg.libXdmcp
            pkgs.xorg.libXext
            pkgs.xorg.libXrender
            pkgs.xorg.libXt
            pkgs.xorg.libSM
            pkgs.xorg.libICE
            pkgs.xorg.libxcb
            pkgs.xorg.xcbutil
            pkgs.xorg.xcbutilcursor
            pkgs.xorg.xcbutilimage
            pkgs.xorg.xcbutilkeysyms
            pkgs.xorg.xcbutilrenderutil
            pkgs.xorg.xcbutilwm
            pkgs.libGL
            pkgs.libGLU
            pkgs.libxkbcommon
            pkgs.mesa
          ];
          hasPy = name: builtins.hasAttr name py;
          pyOpt = name: if hasPy name then [ (builtins.getAttr name py) ] else [ ];
          mkFenicsDolfinx = petsc4pyPkg: (py."fenics-dolfinx".override {
            petsc4py = petsc4pyPkg;
          }).overridePythonAttrs (
            old: {
              nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ py.cmake ];
              doCheck = false;
              doInstallCheck = false;
            }
          );
          fenicsDolfinx = mkFenicsDolfinx py.petsc4py;
          petsc4pyComplex = py.petsc4py.override {
            scalarType = "complex";
            withHypre = false;
          };
          petsc4pyComplexSingle = py.petsc4py.override {
            scalarType = "complex";
            precision = "single";
            withHypre = false;
            withSuperLuDist = false;
            withFftw = false;
            withSuitesparse = false;
          };
          fenicsDolfinxComplex = mkFenicsDolfinx petsc4pyComplex;
          fenicsDolfinxComplexSingle = mkFenicsDolfinx petsc4pyComplexSingle;

          linuxCudaSupported = system == "x86_64-linux";
          pkgsCuda = if linuxCudaSupported then import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              cudaSupport = true;
            };
            overlays = [
              (_final: _prev: {
                # WSL2 single-node CUDA shells do not require CUDA-aware MPI.
                # Reuse the stable CPU MPI/UCX/UCC stack so PETSc/DOLFINx CUDA can
                # build without pulling in the currently failing CUDA-UCX closure.
                ucx = pkgs.ucx;
                ucc = pkgs.ucc;
                openmpi = pkgs.openmpi;
              })
            ];
          } else null;
          pythonCuda = if linuxCudaSupported then pkgsCuda.python313 else null;
          pyCuda = if linuxCudaSupported then pythonCuda.pkgs else null;
          hasCudaPy = name: linuxCudaSupported && builtins.hasAttr name pyCuda;
          pyCudaOpt = name: if hasCudaPy name then [ (builtins.getAttr name pyCuda) ] else [ ];
          mkCudaPetsc = { scalarType ? null, precision ? null }:
            if linuxCudaSupported then
              (pkgsCuda.petsc.override ({
                mpi = pkgsCuda.openmpi;
                python3Packages = pyCuda;
                pythonSupport = true;
              } // lib.optionalAttrs (scalarType != null) {
                inherit scalarType;
                withHypre = false;
                withSuperLuDist = false;
                withFftw = false;
                withSuitesparse = false;
              } // lib.optionalAttrs (precision != null) {
                inherit precision;
              })).overrideAttrs (old: {
                nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pkgsCuda.cudaPackages.cuda_nvcc ];
                buildInputs = (old.buildInputs or [ ]) ++ [
                  pkgsCuda.cudaPackages.cuda_cudart
                  pkgsCuda.cudaPackages.libcublas
                  pkgsCuda.cudaPackages.libcusolver
                  pkgsCuda.cudaPackages.libcusparse
                ];
                configureFlags = (old.configureFlags or [ ]) ++ [
                  "--with-cuda=1"
                  "--with-cudac=${pkgsCuda.cudaPackages.cuda_nvcc}/bin/nvcc"
                  "--with-cuda-dir=${pkgsCuda.cudaPackages.cudatoolkit}"
                  "--with-cublas=1"
                  "--with-cusparse=1"
                  "--with-cusolver=1"
                ];
                doInstallCheck = false;
                postInstall = lib.replaceStrings [ "--replace-fail" ] [ "--replace" ] (old.postInstall or "");
              })
            else null;
          cudaPetsc = mkCudaPetsc { };
          cudaPetscComplex = mkCudaPetsc { scalarType = "complex"; };
          cudaPetscComplexSingle = mkCudaPetsc { scalarType = "complex"; precision = "single"; };
          cudaPetsc4py = if linuxCudaSupported then pyCuda.toPythonModule cudaPetsc else null;
          cudaPetscComplex4py = if linuxCudaSupported then pyCuda.toPythonModule cudaPetscComplex else null;
          cudaPetscComplexSingle4py = if linuxCudaSupported then pyCuda.toPythonModule cudaPetscComplexSingle else null;

          mkCudaSlepc = petscPkg: if linuxCudaSupported then (
            pkgsCuda.callPackage "${nixpkgsPath}/pkgs/by-name/sl/slepc/package.nix" {
              python3Packages = pyCuda;
              petsc = petscPkg;
              pythonSupport = true;
            }
          ).overrideAttrs (old: {
            doInstallCheck = false;
            doCheck = false;
          }) else null;
          cudaSlepc = mkCudaSlepc cudaPetsc;
          cudaSlepcComplex = mkCudaSlepc cudaPetscComplex;
          cudaSlepcComplexSingle = mkCudaSlepc cudaPetscComplexSingle;
          cudaSlepc4py = if linuxCudaSupported then pyCuda.toPythonModule cudaSlepc else null;
          cudaSlepcComplex4py = if linuxCudaSupported then pyCuda.toPythonModule cudaSlepcComplex else null;
          cudaSlepcComplexSingle4py = if linuxCudaSupported then pyCuda.toPythonModule cudaSlepcComplexSingle else null;

          mkCudaDolfinx = petscPkg: slepcPkg:
            if linuxCudaSupported then pkgsCuda.callPackage "${nixpkgsPath}/pkgs/by-name/do/dolfinx/package.nix" {
              python3Packages = pyCuda;
              petsc = petscPkg;
              slepc = slepcPkg;
            } else null;
          cudaDolfinx = mkCudaDolfinx cudaPetsc cudaSlepc;
          cudaDolfinxComplex = mkCudaDolfinx cudaPetscComplex cudaSlepcComplex;
          cudaDolfinxComplexSingle = mkCudaDolfinx cudaPetscComplexSingle cudaSlepcComplexSingle;

          mkCudaFenicsDolfinx = dolfinxPkg: petsc4pyPkg: slepc4pyPkg:
            if linuxCudaSupported then (
              pyCuda.callPackage "${nixpkgsPath}/pkgs/development/python-modules/fenics-dolfinx/default.nix" {
                dolfinx = dolfinxPkg;
                petsc4py = petsc4pyPkg;
                slepc4py = slepc4pyPkg;
              }
            ).overridePythonAttrs (
              old: {
                nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pyCuda.cmake ];
                doCheck = false;
                doInstallCheck = false;
              }
            ) else null;
          cudaFenicsDolfinx = mkCudaFenicsDolfinx cudaDolfinx cudaPetsc4py cudaSlepc4py;
          cudaFenicsDolfinxComplex = mkCudaFenicsDolfinx cudaDolfinxComplex cudaPetscComplex4py cudaSlepcComplex4py;
          cudaFenicsDolfinxComplexSingle = mkCudaFenicsDolfinx cudaDolfinxComplexSingle cudaPetscComplexSingle4py cudaSlepcComplexSingle4py;

          mkShellHook = {
            pkgsFor,
            pythonFor,
            envProfile,
            venvDir,
            extraLinuxRuntimeLibs ? [ ],
            extraLinuxLibraryPath ? "",
            extraPrelude ? "",
          }:
            ''
              export UV_PYTHON="${pythonFor}/bin/python3"
              export UV_PYTHON_PREFERENCE=only-system
              export PYTHONNOUSERSITE=1
              export HDF5_DIR="${pkgsFor.hdf5}"
              export PYEIDORS_ENV_PROFILE="${envProfile}"
              export PYEIDORS_ACTIVE_VENV="${venvDir}"
              export PYEIDORS_ENV_SYNC_INEXACT="''${PYEIDORS_ENV_SYNC_INEXACT:-1}"
              ${extraPrelude}

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

              if [ "$(uname -s)" = "Linux" ]; then
                export LD_LIBRARY_PATH="${lib.makeLibraryPath ([ pkgsFor.stdenv.cc.cc pkgsFor.zlib pkgsFor.zstd ] ++ extraLinuxRuntimeLibs)}${extraLinuxLibraryPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              fi

              nix_python_mm="$($UV_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

              recreate_venv=0
              if [ ! -d "$PYEIDORS_ACTIVE_VENV" ]; then
                recreate_venv=1
              elif [ -x "$PYEIDORS_ACTIVE_VENV/bin/python" ]; then
                venv_python_mm="$($PYEIDORS_ACTIVE_VENV/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
                if [ -z "$venv_python_mm" ] || [ "$venv_python_mm" != "$nix_python_mm" ]; then
                  echo "[nix+uv] Rebuilding $PYEIDORS_ACTIVE_VENV because Python version changed ($PYEIDORS_ACTIVE_VENV=$venv_python_mm, nix=$nix_python_mm)."
                  recreate_venv=1
                fi
              else
                recreate_venv=1
              fi

              if [ "$recreate_venv" -eq 1 ]; then
                rm -rf "$PYEIDORS_ACTIVE_VENV"
                echo "[nix+uv] Creating $PYEIDORS_ACTIVE_VENV with access to Nix site-packages..."
                uv venv --python "$UV_PYTHON" --system-site-packages "$PYEIDORS_ACTIVE_VENV"
              fi

              source "$PYEIDORS_ACTIVE_VENV/bin/activate"

              if [ -f scripts/env/cache_session.sh ]; then
                # shellcheck disable=SC1091
                source scripts/env/cache_session.sh
                pyeidors_cache_session_init ".pyeidors_cache/v2"
              fi

              venv_site="$($PYEIDORS_ACTIVE_VENV/bin/python - <<'PY'
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

              if [ -z "''${PYEIDORS_SHELL_HOOK_READY:-}" ]; then
                if [ -x scripts/env/sync_locked_env.sh ]; then
                  echo "[nix+uv] Checking locked Python environment profile (torch+cuqi+dev+eit-app)..."
                  if ! scripts/env/sync_locked_env.sh --check; then
                    if [ "''${PYEIDORS_GUI_LAUNCH:-0}" = "1" ]; then
                      echo "[nix+uv] Refreshing locked Python environment profile..."
                    else
                      echo "[nix+uv] Drift detected. Attempting automatic repair..."
                    fi
                    if ! scripts/env/sync_locked_env.sh --repair; then
                      echo "[nix+uv] ERROR: environment repair failed."
                      echo "[nix+uv] Manual repair command: scripts/env/sync_locked_env.sh --repair"
                      exit 1
                    fi
                  fi
                else
                  echo "[nix+uv] WARNING: scripts/env/sync_locked_env.sh not found; skipping env sync."
                fi

                if [ "''${ENABLE_PERFORMANCE_EXTRAS:-0}" = "1" ]; then
                  perf_status="$($UV_PYTHON - <<'PY'
import importlib

status = {}
for name in ("pyamg", "sksparse"):
    try:
        importlib.import_module(name)
        status[name] = "available"
    except Exception:
        status[name] = "missing"

cholmod = "missing"
if status["sksparse"] == "available":
    try:
        from sksparse import cholmod as _cholmod  # noqa: F401
        cholmod = "available"
    except Exception:
        cholmod = "missing"

print(
    f"[nix+uv] Optional performance extras status: "
    f"pyamg={status['pyamg']}, sksparse={status['sksparse']}, cholmod={cholmod}"
    + " (missing extras do not block the core environment)"
)
PY
)"
                  echo "$perf_status"
                fi

                if [ "$PYEIDORS_ENV_PROFILE" = "cuda" ] || [ "$PYEIDORS_ENV_PROFILE" = "complex-cuda" ] || [ "$PYEIDORS_ENV_PROFILE" = "complex64-cuda" ]; then
                  echo "[nix+uv] CUDA profile ready. Verify PETSc CUDA backend with:"
                  echo "  python scripts/diagnostics/probe_petsc_cuda.py --require cuda --pretty"
                fi

                if [ "$PYEIDORS_ENV_PROFILE" = "complex" ] || [ "$PYEIDORS_ENV_PROFILE" = "complex64" ] || [ "$PYEIDORS_ENV_PROFILE" = "complex-cuda" ] || [ "$PYEIDORS_ENV_PROFILE" = "complex64-cuda" ]; then
                  echo "[nix+uv] Complex PETSc profile ready. Verify scalar type with:"
                  echo "  python - <<'PY'"
                  echo "from petsc4py import PETSc; import numpy as np; print(np.dtype(PETSc.ScalarType))"
                  echo "PY"
                fi

                echo "[nix+uv] Dev shell ready ($PYEIDORS_ENV_PROFILE)."
                echo "[nix+uv] Core dependency import checks completed during shell entry."
                export PYEIDORS_SHELL_HOOK_READY=1
              fi
            '';
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
              pkgs.suitesparse
              pkgs.zstd
              pkgs.glib
              pkgs.dbus
              pkgs.fontconfig
              pkgs.freetype
              pkgs.liberation_ttf
              pkgs.expat

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
            ] ++ pyOpt "pyamg" ++ pyOpt "scikit-sparse" ++ pyOpt "scikitsparse" ++ [
              py.pytest
              py."pytest-cov"
              py.black
              py.flake8
              pkgs.pre-commit
            ];

            shellHook = mkShellHook {
              pkgsFor = pkgs;
              pythonFor = python;
              envProfile = "default";
              venvDir = ".venv";
              extraLinuxRuntimeLibs = linuxGuiLibs;
              extraLinuxLibraryPath = ":/usr/lib/wsl/lib";
              extraPrelude = ''
                export LIBGL_DRIVERS_PATH="${pkgs.mesa}/lib/dri"
                if [ -d /usr/lib/wsl/lib ]; then
                  export PATH="/usr/lib/wsl/lib:$PATH"
                fi
              '';
            };
          };

          complex = pkgs.mkShell {
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
              pkgs.suitesparse
              pkgs.zstd
              pkgs.glib
              pkgs.dbus
              pkgs.fontconfig
              pkgs.freetype
              pkgs.liberation_ttf
              pkgs.expat

              petsc4pyComplex
              fenicsDolfinxComplex
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
            ] ++ pyOpt "pyamg" ++ pyOpt "scikit-sparse" ++ pyOpt "scikitsparse" ++ [
              py.pytest
              py."pytest-cov"
              py.black
              py.flake8
              pkgs.pre-commit
            ];

            shellHook = mkShellHook {
              pkgsFor = pkgs;
              pythonFor = python;
              envProfile = "complex";
              venvDir = ".venv-complex";
              extraLinuxRuntimeLibs = linuxGuiLibs;
              extraLinuxLibraryPath = ":/usr/lib/wsl/lib";
              extraPrelude = ''
                export PYEIDORS_PETSC_SCALAR_TYPE="complex"
                export LIBGL_DRIVERS_PATH="${pkgs.mesa}/lib/dri"
                if [ -d /usr/lib/wsl/lib ]; then
                  export PATH="/usr/lib/wsl/lib:$PATH"
                fi
              '';
            };
          };

        }
        // lib.optionalAttrs linuxCudaSupported {
          cuda = pkgsCuda.mkShell {
            packages = [
              pkgsCuda.uv
              pythonCuda
              pkgsCuda.openmpi
              pkgsCuda.hdf5
              pkgsCuda.gmsh
              pkgsCuda.pkg-config
              pkgsCuda.cmake
              pkgsCuda.ninja
              pkgsCuda.gfortran
              pkgsCuda.openblas
              pkgsCuda.suitesparse
              pkgsCuda.zstd
              pkgsCuda.glib
              pkgsCuda.dbus
              pkgsCuda.fontconfig
              pkgsCuda.freetype
              pkgsCuda.liberation_ttf
              pkgsCuda.expat
              pkgsCuda.cudaPackages.cuda_nvcc
              pkgsCuda.cudaPackages.cudatoolkit
              pkgsCuda.cudaPackages.cuda_cudart
              pkgsCuda.cudaPackages.libcublas
              pkgsCuda.cudaPackages.libcusolver
              pkgsCuda.cudaPackages.libcusparse
              pkgsCuda.cudaPackages.libnvjitlink

              cudaPetsc
              cudaPetsc4py
              cudaSlepc
              cudaSlepc4py
              cudaFenicsDolfinx
              pyCuda."fenics-basix"
              pyCuda."fenics-ffcx"
              pyCuda."fenics-ufl"
              pyCuda.mpi4py

              pyCuda.numpy
              pyCuda.scipy
              pyCuda.matplotlib
              pyCuda.pandas
              pyCuda.h5py
              pyCuda.pyyaml
              pyCuda.meshio
              pyCuda.gmsh
            ] ++ pyCudaOpt "pyamg" ++ pyCudaOpt "scikit-sparse" ++ pyCudaOpt "scikitsparse" ++ [
              pyCuda.pytest
              pyCuda."pytest-cov"
              pyCuda.black
              pyCuda.flake8
              pkgsCuda.pre-commit
            ];

            shellHook = mkShellHook {
              pkgsFor = pkgsCuda;
              pythonFor = pythonCuda;
              envProfile = "cuda";
              venvDir = ".venv-cuda";
              extraLinuxRuntimeLibs = linuxGuiLibs ++ [
                pkgsCuda.cudaPackages.cuda_cudart
                pkgsCuda.cudaPackages.libcublas
                pkgsCuda.cudaPackages.libcusolver
                pkgsCuda.cudaPackages.libcusparse
                pkgsCuda.cudaPackages.libnvjitlink
              ];
              extraLinuxLibraryPath = ":/usr/lib/wsl/lib";
              extraPrelude = ''
                export CUDA_HOME="${pkgsCuda.cudaPackages.cudatoolkit}"
                export CUDA_PATH="$CUDA_HOME"
                export CUDACXX="${pkgsCuda.cudaPackages.cuda_nvcc}/bin/nvcc"
                export PETSC_DIR="${cudaPetsc}"
                export SLEPC_DIR="${cudaSlepc}"
                export PYEIDORS_PETSC_DEVICE_DEFAULT="cuda"
                export PETSC_OPTIONS="-use_gpu_aware_mpi 0 -nox_warning''${PETSC_OPTIONS:+ $PETSC_OPTIONS}"
                if [ -d /usr/lib/wsl/lib ]; then
                  export PATH="/usr/lib/wsl/lib:$PATH"
                fi
              '';
            };
          };

          "complex-cuda" = pkgsCuda.mkShell {
            packages = [
              pkgsCuda.uv
              pythonCuda
              pkgsCuda.openmpi
              pkgsCuda.hdf5
              pkgsCuda.gmsh
              pkgsCuda.pkg-config
              pkgsCuda.cmake
              pkgsCuda.ninja
              pkgsCuda.gfortran
              pkgsCuda.openblas
              pkgsCuda.suitesparse
              pkgsCuda.zstd
              pkgsCuda.glib
              pkgsCuda.dbus
              pkgsCuda.fontconfig
              pkgsCuda.freetype
              pkgsCuda.liberation_ttf
              pkgsCuda.expat
              pkgsCuda.cudaPackages.cuda_nvcc
              pkgsCuda.cudaPackages.cudatoolkit
              pkgsCuda.cudaPackages.cuda_cudart
              pkgsCuda.cudaPackages.libcublas
              pkgsCuda.cudaPackages.libcusolver
              pkgsCuda.cudaPackages.libcusparse
              pkgsCuda.cudaPackages.libnvjitlink

              cudaPetscComplex
              cudaPetscComplex4py
              cudaSlepcComplex
              cudaSlepcComplex4py
              cudaFenicsDolfinxComplex
              pyCuda."fenics-basix"
              pyCuda."fenics-ffcx"
              pyCuda."fenics-ufl"
              pyCuda.mpi4py

              pyCuda.numpy
              pyCuda.scipy
              pyCuda.matplotlib
              pyCuda.pandas
              pyCuda.h5py
              pyCuda.pyyaml
              pyCuda.meshio
              pyCuda.gmsh
            ] ++ pyCudaOpt "pyamg" ++ pyCudaOpt "scikit-sparse" ++ pyCudaOpt "scikitsparse" ++ [
              pyCuda.pytest
              pyCuda."pytest-cov"
              pyCuda.black
              pyCuda.flake8
              pkgsCuda.pre-commit
            ];

            shellHook = mkShellHook {
              pkgsFor = pkgsCuda;
              pythonFor = pythonCuda;
              envProfile = "complex-cuda";
              venvDir = ".venv-complex-cuda";
              extraLinuxRuntimeLibs = linuxGuiLibs ++ [
                pkgsCuda.cudaPackages.cuda_cudart
                pkgsCuda.cudaPackages.libcublas
                pkgsCuda.cudaPackages.libcusolver
                pkgsCuda.cudaPackages.libcusparse
                pkgsCuda.cudaPackages.libnvjitlink
              ];
              extraLinuxLibraryPath = ":/usr/lib/wsl/lib";
              extraPrelude = ''
                export PYEIDORS_PETSC_SCALAR_TYPE="complex"
                export CUDA_HOME="${pkgsCuda.cudaPackages.cudatoolkit}"
                export CUDA_PATH="$CUDA_HOME"
                export CUDACXX="${pkgsCuda.cudaPackages.cuda_nvcc}/bin/nvcc"
                export PETSC_DIR="${cudaPetscComplex}"
                export SLEPC_DIR="${cudaSlepcComplex}"
                export PYEIDORS_PETSC_DEVICE_DEFAULT="cuda"
                export PETSC_OPTIONS="-use_gpu_aware_mpi 0 -nox_warning''${PETSC_OPTIONS:+ $PETSC_OPTIONS}"
                if [ -d /usr/lib/wsl/lib ]; then
                  export PATH="/usr/lib/wsl/lib:$PATH"
                fi
              '';
            };
          };

          "complex64-cuda" = pkgsCuda.mkShell {
            packages = [
              pkgsCuda.uv
              pythonCuda
              pkgsCuda.openmpi
              pkgsCuda.hdf5
              pkgsCuda.gmsh
              pkgsCuda.pkg-config
              pkgsCuda.cmake
              pkgsCuda.ninja
              pkgsCuda.gfortran
              pkgsCuda.openblas
              pkgsCuda.suitesparse
              pkgsCuda.zstd
              pkgsCuda.glib
              pkgsCuda.dbus
              pkgsCuda.fontconfig
              pkgsCuda.freetype
              pkgsCuda.liberation_ttf
              pkgsCuda.expat
              pkgsCuda.cudaPackages.cuda_nvcc
              pkgsCuda.cudaPackages.cudatoolkit
              pkgsCuda.cudaPackages.cuda_cudart
              pkgsCuda.cudaPackages.libcublas
              pkgsCuda.cudaPackages.libcusolver
              pkgsCuda.cudaPackages.libcusparse
              pkgsCuda.cudaPackages.libnvjitlink

              cudaPetscComplexSingle
              cudaPetscComplexSingle4py
              cudaSlepcComplexSingle
              cudaSlepcComplexSingle4py
              cudaFenicsDolfinxComplexSingle
              pyCuda."fenics-basix"
              pyCuda."fenics-ffcx"
              pyCuda."fenics-ufl"
              pyCuda.mpi4py

              pyCuda.numpy
              pyCuda.scipy
              pyCuda.matplotlib
              pyCuda.pandas
              pyCuda.h5py
              pyCuda.pyyaml
              pyCuda.meshio
              pyCuda.gmsh
            ] ++ pyCudaOpt "pyamg" ++ pyCudaOpt "scikit-sparse" ++ pyCudaOpt "scikitsparse" ++ [
              pyCuda.pytest
              pyCuda."pytest-cov"
              pyCuda.black
              pyCuda.flake8
              pkgsCuda.pre-commit
            ];

            shellHook = mkShellHook {
              pkgsFor = pkgsCuda;
              pythonFor = pythonCuda;
              envProfile = "complex64-cuda";
              venvDir = ".venv-complex64-cuda";
              extraLinuxRuntimeLibs = linuxGuiLibs ++ [
                pkgsCuda.cudaPackages.cuda_cudart
                pkgsCuda.cudaPackages.libcublas
                pkgsCuda.cudaPackages.libcusolver
                pkgsCuda.cudaPackages.libcusparse
                pkgsCuda.cudaPackages.libnvjitlink
              ];
              extraLinuxLibraryPath = ":/usr/lib/wsl/lib";
              extraPrelude = ''
                export PYEIDORS_PETSC_SCALAR_TYPE="complex64"
                export CUDA_HOME="${pkgsCuda.cudaPackages.cudatoolkit}"
                export CUDA_PATH="$CUDA_HOME"
                export CUDACXX="${pkgsCuda.cudaPackages.cuda_nvcc}/bin/nvcc"
                export PETSC_DIR="${cudaPetscComplexSingle}"
                export SLEPC_DIR="${cudaSlepcComplexSingle}"
                export PYEIDORS_PETSC_DEVICE_DEFAULT="cuda"
                export PETSC_OPTIONS="-use_gpu_aware_mpi 0 -nox_warning''${PETSC_OPTIONS:+ $PETSC_OPTIONS}"
                if [ -d /usr/lib/wsl/lib ]; then
                  export PATH="/usr/lib/wsl/lib:$PATH"
                fi
              '';
            };
          };

          complex64 = pkgs.mkShell {
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
              pkgs.suitesparse
              pkgs.zstd
              pkgs.glib
              pkgs.dbus
              pkgs.fontconfig
              pkgs.freetype
              pkgs.liberation_ttf
              pkgs.expat

              petsc4pyComplexSingle
              fenicsDolfinxComplexSingle
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
            ] ++ pyOpt "pyamg" ++ pyOpt "scikit-sparse" ++ pyOpt "scikitsparse" ++ [
              py.pytest
              py."pytest-cov"
              py.black
              py.flake8
              pkgs.pre-commit
            ];

            shellHook = mkShellHook {
              pkgsFor = pkgs;
              pythonFor = python;
              envProfile = "complex64";
              venvDir = ".venv-complex64";
              extraLinuxRuntimeLibs = linuxGuiLibs;
              extraLinuxLibraryPath = ":/usr/lib/wsl/lib";
              extraPrelude = ''
                export PYEIDORS_PETSC_SCALAR_TYPE="complex64"
                export LIBGL_DRIVERS_PATH="${pkgs.mesa}/lib/dri"
                if [ -d /usr/lib/wsl/lib ]; then
                  export PATH="/usr/lib/wsl/lib:$PATH"
                fi
              '';
            };
          };
        }
      );
    };
}

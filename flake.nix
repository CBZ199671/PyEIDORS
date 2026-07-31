{
  description = "PyEIDORS pure Nix development shells with FEniCSx (DOLFINx)";

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
      amgxGitCommit = "4d1bda0016c42bbe9c0470ca976f10cf6774fd8a";
      amgxGitUrl = "https://github.com/NVIDIA/AMGX.git";
      amgxSourceHash = "sha256-XKyGG1wsG37qlSTukZMl8BKyi248SCQKHdlgVYfnR6A=";
    in
    {
      packages = forAllSystems (
        system:
        let
          nixpkgsPath = nixpkgs.outPath;
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python313;
          py = python.pkgs;
          hasPy = pyFor: name: builtins.hasAttr name pyFor;
          pyOpt = pyFor: name: if hasPy pyFor name then [ (builtins.getAttr name pyFor) ] else [ ];
          pyeidorsVersion = (builtins.fromTOML (builtins.readFile ./pyproject.toml)).project.version;
          pyeidorsSource = lib.cleanSourceWith {
            src = ./.;
            filter =
              path: type:
              let
                root = toString ./.;
                rel = lib.removePrefix "${root}/" (toString path);
              in
              rel == "pyproject.toml"
              || rel == "pyeidors.backend.json"
              || rel == "README.md"
              || rel == "LICENSE"
              || rel == "src"
              || (
                lib.hasPrefix "src/" rel
                && !(lib.hasInfix "/__pycache__/" rel)
                && !(lib.hasSuffix ".pyc" rel)
                && !(lib.hasSuffix ".pyo" rel)
                && !(lib.hasPrefix "src/pyeidors.egg-info/" rel)
                && !(lib.hasPrefix "src/hello." rel)
              );
          };
          mkLinuxGuiLibs = pkgsFor: [
            pkgsFor.glib
            pkgsFor.dbus
            pkgsFor.wayland
            pkgsFor.fontconfig
            pkgsFor.freetype
            pkgsFor.expat
            pkgsFor.xorg.libX11
            pkgsFor.xorg.libXau
            pkgsFor.xorg.libXdmcp
            pkgsFor.xorg.libXext
            pkgsFor.xorg.libXrender
            pkgsFor.xorg.libXt
            pkgsFor.xorg.libSM
            pkgsFor.xorg.libICE
            pkgsFor.xorg.libxcb
            pkgsFor.xorg.xcbutil
            pkgsFor.xorg.xcbutilcursor
            pkgsFor.xorg.xcbutilimage
            pkgsFor.xorg.xcbutilkeysyms
            pkgsFor.xorg.xcbutilrenderutil
            pkgsFor.xorg.xcbutilwm
            pkgsFor.libGL
            pkgsFor.libGLU
            pkgsFor.libxkbcommon
            pkgsFor.mesa
          ];
          mkFenicsDolfinx = pyFor: petsc4pyPkg: (pyFor."fenics-dolfinx".override {
            petsc4py = petsc4pyPkg;
          }).overridePythonAttrs (
            old: {
              nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pyFor.cmake ];
              doCheck = false;
              doInstallCheck = false;
            }
          );
          fenicsDolfinx = mkFenicsDolfinx py py.petsc4py;
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
          fenicsDolfinxComplex = mkFenicsDolfinx py petsc4pyComplex;
          fenicsDolfinxComplexSingle = mkFenicsDolfinx py petsc4pyComplexSingle;

          linuxCudaSupported = system == "x86_64-linux";
          cudaMpiOverlay = (_final: _prev: {
            ucx = pkgs.ucx;
            ucc = pkgs.ucc;
            openmpi = pkgs.openmpi;
          });
          mkCudaPkgs = {
            configOverrides ? { },
            extraOverlays ? [ ],
          }: import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              cudaSupport = true;
            } // configOverrides;
            overlays = [ cudaMpiOverlay ] ++ extraOverlays;
          };
          pkgsCuda = if linuxCudaSupported then mkCudaPkgs { } else null;
          pkgsCudaSm61 = if linuxCudaSupported then mkCudaPkgs {
            configOverrides = {
              cudaCapabilities = [ "6.1" ];
              cudaForwardCompat = false;
            };
            extraOverlays = [
              (_final: prev: {
                cudaPackages = prev.cudaPackages.overrideScope (_cudaFinal: cudaPrev: {
                  cudnn = cudaPrev.cudnn.overrideAttrs (old: {
                    meta = (old.meta or { }) // {
                      badPlatforms = [ ];
                    };
                  });
                });
              })
            ];
          } else null;
          pythonCuda = if linuxCudaSupported then pkgsCuda.python313 else null;
          pyCuda = if linuxCudaSupported then pythonCuda.pkgs else null;
          pythonCudaSm61 = if linuxCudaSupported then pkgsCudaSm61.python313 else null;
          pyCudaSm61 = if linuxCudaSupported then pythonCudaSm61.pkgs else null;
          amgxSource = if linuxCudaSupported then pkgsCuda.fetchgit {
            url = amgxGitUrl;
            rev = amgxGitCommit;
            hash = amgxSourceHash;
            fetchSubmodules = true;
          } else null;
          amgxSourceArchive = if linuxCudaSupported then pkgsCuda.runCommand
            "AMGX-${lib.substring 0 7 amgxGitCommit}.tar.gz"
            { nativeBuildInputs = [ pkgsCuda.gnutar pkgsCuda.gzip ]; }
            ''
              cp -R --no-preserve=mode,ownership ${amgxSource} "$TMPDIR/AMGX-${amgxGitCommit}"
              chmod -R u+rwX,go+rX "$TMPDIR/AMGX-${amgxGitCommit}"
              cp ${pkgsCuda.cudaPackages.cuda_nvtx.include}/include/nvToolsExt.h \
                "$TMPDIR/AMGX-${amgxGitCommit}/include/nvToolsExt.h"
              cp ${pkgsCuda.cudaPackages.libcurand.include}/include/curand*.h \
                "$TMPDIR/AMGX-${amgxGitCommit}/include/"
              substituteInPlace "$TMPDIR/AMGX-${amgxGitCommit}/CMakeLists.txt" \
                --replace-fail 'target_link_libraries(amgx CUDA::cublas CUDA::cusparse CUDA::cusolver CUDA::nvToolsExt m pthread)' \
                  'target_link_libraries(amgx CUDA::cublas CUDA::cusparse CUDA::cusolver CUDA::nvToolsExt m pthread "${pkgsCuda.openmpi}/lib/libmpi.so")' \
                --replace-fail 'target_link_libraries(amgxsh CUDA::cublas CUDA::cusparse CUDA::cusolver CUDA::nvToolsExt m pthread)' \
                  'target_link_libraries(amgxsh CUDA::cublas CUDA::cusparse CUDA::cusolver CUDA::nvToolsExt m pthread "${pkgsCuda.openmpi}/lib/libmpi.so")'
              substituteInPlace "$TMPDIR/AMGX-${amgxGitCommit}/src/CMakeLists.txt" \
                --replace-fail 'target_link_libraries(amgx_tests_launcher amgxsh ''${libs_all} OpenMP::OpenMP_C)' \
                  'target_link_libraries(amgx_tests_launcher amgxsh ''${libs_all} OpenMP::OpenMP_C "${pkgsCuda.openmpi}/lib/libmpi.so")'
              tar -C "$TMPDIR" \
                --owner=0 \
                --group=0 \
                --numeric-owner \
                -czf "$out" AMGX-${amgxGitCommit}
            ''
          else null;
          cmakeForPetscExternalPackages = if linuxCudaSupported then pkgsCuda.runCommand "cmake-for-petsc-external-packages" { } ''
            mkdir -p "$out/bin"
            for tool in cmake ctest cpack; do
              ln -s "${pkgsCuda.cmake}/bin/$tool" "$out/bin/$tool"
            done
          '' else null;
          mkCudaPetsc = {
            pkgsCudaFor ? pkgsCuda,
            pyCudaFor ? pyCuda,
            amgxSourceArchiveFor ? amgxSourceArchive,
            cmakeForPetscExternalPackagesFor ? cmakeForPetscExternalPackages,
            scalarType ? null,
            precision ? null,
            withAmgx ? false,
            allowComplexAmgx ? false,
          }:
            assert !withAmgx || allowComplexAmgx || scalarType == null || scalarType == "real";
            assert !withAmgx || allowComplexAmgx || precision == null || precision == "double";
            if linuxCudaSupported then
              (pkgsCudaFor.petsc.override ({
                mpi = pkgsCudaFor.openmpi;
                python3Packages = pyCudaFor;
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
                postPatch = (old.postPatch or "") + lib.optionalString (withAmgx && allowComplexAmgx) ''
                  substituteInPlace src/ksp/pc/impls/amgx/amgx.cxx \
                    --replace-fail '  PetscScalar      strength_threshold;' '  PetscReal        strength_threshold;' \
                    --replace-fail '  PetscScalar jacobi_relaxation_factor;' '  PetscReal jacobi_relaxation_factor;' \
                    --replace-fail '  PetscScalar gs_symmetric;' '  PetscReal gs_symmetric;' \
                    --replace-fail '    PetscCall(PetscOptionsScalar("-pc_amgx_jacobi_relaxation_factor", "AmgX AMG Jacobi Relaxation Factor", "", amgx->jacobi_relaxation_factor, &amgx->jacobi_relaxation_factor, NULL));' '    PetscCall(PetscOptionsReal("-pc_amgx_jacobi_relaxation_factor", "AmgX AMG Jacobi Relaxation Factor", "", amgx->jacobi_relaxation_factor, &amgx->jacobi_relaxation_factor, NULL));' \
                    --replace-fail '    PetscCall(PetscOptionsScalar("-pc_amgx_gs_symmetric", "AmgX AMG Gauss Seidel Symmetric", "", amgx->gs_symmetric, &amgx->gs_symmetric, NULL));' '    PetscCall(PetscOptionsReal("-pc_amgx_gs_symmetric", "AmgX AMG Gauss Seidel Symmetric", "", amgx->gs_symmetric, &amgx->gs_symmetric, NULL));' \
                    --replace-fail '  PetscCall(PetscOptionsScalar("-pc_amgx_strength_threshold", "AmgX AMG Strength Threshold", "", amgx->strength_threshold, &amgx->strength_threshold, NULL));' '  PetscCall(PetscOptionsReal("-pc_amgx_strength_threshold", "AmgX AMG Strength Threshold", "", amgx->strength_threshold, &amgx->strength_threshold, NULL));'
                  substituteInPlace src/ksp/pc/impls/amgx/amgx.cxx \
                    --replace-fail '    PetscCallAmgX(AMGX_matrix_create(&amgx->A, amgx->rsrc, AMGX_mode_dDDI));' '#if defined(PETSC_USE_COMPLEX) && defined(PETSC_USE_REAL_SINGLE)
                    const AMGX_Mode amgxMode = AMGX_mode_dCCI;
                  #elif defined(PETSC_USE_COMPLEX)
                    const AMGX_Mode amgxMode = AMGX_mode_dZZI;
                  #elif defined(PETSC_USE_REAL_SINGLE)
                    const AMGX_Mode amgxMode = AMGX_mode_dFFI;
                  #else
                    const AMGX_Mode amgxMode = AMGX_mode_dDDI;
                  #endif
                    PetscCallAmgX(AMGX_matrix_create(&amgx->A, amgx->rsrc, amgxMode));' \
                    --replace-fail '    PetscCallAmgX(AMGX_vector_create(&amgx->sol, amgx->rsrc, AMGX_mode_dDDI));' '    PetscCallAmgX(AMGX_vector_create(&amgx->sol, amgx->rsrc, amgxMode));' \
                    --replace-fail '    PetscCallAmgX(AMGX_vector_create(&amgx->rhs, amgx->rsrc, AMGX_mode_dDDI));' '    PetscCallAmgX(AMGX_vector_create(&amgx->rhs, amgx->rsrc, amgxMode));' \
                    --replace-fail '    PetscCallAmgX(AMGX_solver_create(&amgx->solver, amgx->rsrc, AMGX_mode_dDDI, amgx->cfg));' '    PetscCallAmgX(AMGX_solver_create(&amgx->solver, amgx->rsrc, amgxMode, amgx->cfg));'
                  substituteInPlace src/ksp/pc/impls/amgx/amgx.cxx \
                    --replace-fail '  PetscCheck(AmgXControlMap::Selectors.count(option) == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Selector %s not registered for AmgX.", option);' '  PetscCheck(AmgXControlMap::Selectors.count(option) == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Selector %s not registered for AmgX.", option);
                  amgx->selector = AmgXControlMap::Selectors.at(option);'
                '';
                nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                  pkgsCudaFor.cudaPackages.cuda_nvcc
                ] ++ lib.optional withAmgx cmakeForPetscExternalPackagesFor;
                buildInputs = (old.buildInputs or [ ]) ++ [
                  pkgsCudaFor.cudaPackages.cuda_cudart
                  pkgsCudaFor.cudaPackages.libcublas
                  pkgsCudaFor.cudaPackages.libcusolver
                  pkgsCudaFor.cudaPackages.libcusparse
                ] ++ lib.optionals withAmgx [
                  pkgsCudaFor.cudaPackages.cuda_nvtx
                  pkgsCudaFor.cudaPackages.libcurand
                ];
                configureFlags = (old.configureFlags or [ ]) ++ [
                  "--with-cuda=1"
                  "--with-cudac=${pkgsCudaFor.cudaPackages.cuda_nvcc}/bin/nvcc"
                  "--with-cuda-dir=${pkgsCudaFor.cudaPackages.cudatoolkit}"
                  "--with-cublas=1"
                  "--with-cusparse=1"
                  "--with-cusolver=1"
                ] ++ lib.optionals withAmgx [
                  "--download-amgx=${amgxSourceArchiveFor}"
                  "--with-64-bit-indices=0"
                  "--with-cxx-dialect=17"
                  "--with-cuda-dialect=17"
                ];
                doInstallCheck = false;
                postInstall = lib.replaceStrings [ "--replace-fail" ] [ "--replace" ] (old.postInstall or "");
              })
            else null;
          cudaPetsc = mkCudaPetsc { };
          cudaPetscAmgx = mkCudaPetsc { withAmgx = true; };
          cudaPetscComplex = mkCudaPetsc { scalarType = "complex"; };
          cudaPetscComplexAmgx = mkCudaPetsc {
            scalarType = "complex";
            withAmgx = true;
            allowComplexAmgx = true;
          };
          cudaPetscComplexSingle = mkCudaPetsc { scalarType = "complex"; precision = "single"; };
          cudaPetsc4py = if linuxCudaSupported then pyCuda.toPythonModule cudaPetsc else null;
          cudaPetscAmgx4py = if linuxCudaSupported then pyCuda.toPythonModule cudaPetscAmgx else null;
          cudaPetscComplex4py = if linuxCudaSupported then pyCuda.toPythonModule cudaPetscComplex else null;
          cudaPetscComplexAmgx4py = if linuxCudaSupported then pyCuda.toPythonModule cudaPetscComplexAmgx else null;
          cudaPetscComplexSingle4py = if linuxCudaSupported then pyCuda.toPythonModule cudaPetscComplexSingle else null;
          mkCudaSlepcFor = pkgsCudaFor: pyCudaFor: petscPkg: if linuxCudaSupported then (
            pkgsCudaFor.callPackage "${nixpkgsPath}/pkgs/by-name/sl/slepc/package.nix" {
              python3Packages = pyCudaFor;
              petsc = petscPkg;
              pythonSupport = true;
            }
          ).overrideAttrs (old: {
            doInstallCheck = false;
            doCheck = false;
          }) else null;
          mkCudaSlepc = mkCudaSlepcFor pkgsCuda pyCuda;
          cudaSlepc = mkCudaSlepc cudaPetsc;
          cudaSlepcAmgx = mkCudaSlepc cudaPetscAmgx;
          cudaSlepcComplex = mkCudaSlepc cudaPetscComplex;
          cudaSlepcComplexAmgx = mkCudaSlepc cudaPetscComplexAmgx;
          cudaSlepcComplexSingle = mkCudaSlepc cudaPetscComplexSingle;
          cudaSlepc4py = if linuxCudaSupported then pyCuda.toPythonModule cudaSlepc else null;
          cudaSlepcAmgx4py = if linuxCudaSupported then pyCuda.toPythonModule cudaSlepcAmgx else null;
          cudaSlepcComplex4py = if linuxCudaSupported then pyCuda.toPythonModule cudaSlepcComplex else null;
          cudaSlepcComplexAmgx4py = if linuxCudaSupported then pyCuda.toPythonModule cudaSlepcComplexAmgx else null;
          cudaSlepcComplexSingle4py = if linuxCudaSupported then pyCuda.toPythonModule cudaSlepcComplexSingle else null;
          mkCudaDolfinxFor = pkgsCudaFor: pyCudaFor: petscPkg: slepcPkg:
            if linuxCudaSupported then pkgsCudaFor.callPackage "${nixpkgsPath}/pkgs/by-name/do/dolfinx/package.nix" {
              python3Packages = pyCudaFor;
              petsc = petscPkg;
              slepc = slepcPkg;
            } else null;
          mkCudaDolfinx = mkCudaDolfinxFor pkgsCuda pyCuda;
          cudaDolfinx = mkCudaDolfinx cudaPetsc cudaSlepc;
          cudaDolfinxAmgx = mkCudaDolfinx cudaPetscAmgx cudaSlepcAmgx;
          cudaDolfinxComplex = mkCudaDolfinx cudaPetscComplex cudaSlepcComplex;
          cudaDolfinxComplexAmgx = mkCudaDolfinx cudaPetscComplexAmgx cudaSlepcComplexAmgx;
          cudaDolfinxComplexSingle = mkCudaDolfinx cudaPetscComplexSingle cudaSlepcComplexSingle;
          mkCudaFenicsDolfinxFor = pkgsCudaFor: pyCudaFor: dolfinxPkg: petsc4pyPkg: slepc4pyPkg:
            if linuxCudaSupported then (
              pyCudaFor.callPackage "${nixpkgsPath}/pkgs/development/python-modules/fenics-dolfinx/default.nix" {
                dolfinx = dolfinxPkg;
                petsc4py = petsc4pyPkg;
                slepc4py = slepc4pyPkg;
              }
            ).overridePythonAttrs (
              old: {
                nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pyCudaFor.cmake ];
                doCheck = false;
                doInstallCheck = false;
              }
            ) else null;
          mkCudaFenicsDolfinx = mkCudaFenicsDolfinxFor pkgsCuda pyCuda;
          cudaFenicsDolfinx = mkCudaFenicsDolfinx cudaDolfinx cudaPetsc4py cudaSlepc4py;
          cudaFenicsDolfinxAmgx = mkCudaFenicsDolfinx cudaDolfinxAmgx cudaPetscAmgx4py cudaSlepcAmgx4py;
          cudaFenicsDolfinxComplex = mkCudaFenicsDolfinx cudaDolfinxComplex cudaPetscComplex4py cudaSlepcComplex4py;
          cudaFenicsDolfinxComplexAmgx = mkCudaFenicsDolfinx cudaDolfinxComplexAmgx cudaPetscComplexAmgx4py cudaSlepcComplexAmgx4py;
          cudaFenicsDolfinxComplexSingle = mkCudaFenicsDolfinx cudaDolfinxComplexSingle cudaPetscComplexSingle4py cudaSlepcComplexSingle4py;
          cudaSm61Petsc = mkCudaPetsc {
            pkgsCudaFor = pkgsCudaSm61;
            pyCudaFor = pyCudaSm61;
          };
          cudaSm61PetscComplexSingle = mkCudaPetsc {
            pkgsCudaFor = pkgsCudaSm61;
            pyCudaFor = pyCudaSm61;
            scalarType = "complex";
            precision = "single";
          };
          cudaSm61Petsc4py = if linuxCudaSupported then pyCudaSm61.toPythonModule cudaSm61Petsc else null;
          cudaSm61PetscComplexSingle4py =
            if linuxCudaSupported then pyCudaSm61.toPythonModule cudaSm61PetscComplexSingle else null;
          mkCudaSm61Slepc = mkCudaSlepcFor pkgsCudaSm61 pyCudaSm61;
          cudaSm61Slepc = mkCudaSm61Slepc cudaSm61Petsc;
          cudaSm61SlepcComplexSingle = mkCudaSm61Slepc cudaSm61PetscComplexSingle;
          cudaSm61Slepc4py = if linuxCudaSupported then pyCudaSm61.toPythonModule cudaSm61Slepc else null;
          cudaSm61SlepcComplexSingle4py =
            if linuxCudaSupported then pyCudaSm61.toPythonModule cudaSm61SlepcComplexSingle else null;
          mkCudaSm61Dolfinx = mkCudaDolfinxFor pkgsCudaSm61 pyCudaSm61;
          cudaSm61Dolfinx = mkCudaSm61Dolfinx cudaSm61Petsc cudaSm61Slepc;
          cudaSm61DolfinxComplexSingle =
            mkCudaSm61Dolfinx cudaSm61PetscComplexSingle cudaSm61SlepcComplexSingle;
          mkCudaSm61FenicsDolfinx = mkCudaFenicsDolfinxFor pkgsCudaSm61 pyCudaSm61;
          cudaSm61FenicsDolfinx =
            mkCudaSm61FenicsDolfinx cudaSm61Dolfinx cudaSm61Petsc4py cudaSm61Slepc4py;
          cudaSm61FenicsDolfinxComplexSingle =
            mkCudaSm61FenicsDolfinx
              cudaSm61DolfinxComplexSingle
              cudaSm61PetscComplexSingle4py
              cudaSm61SlepcComplexSingle4py;

          mkRuntimePython = { pkgsFor, pyFor }:
            let
              h5pyMpi = pyFor.h5py.override {
                hdf5 = pkgsFor.hdf5-mpi;
                mpi4py = pyFor.mpi4py;
              };
              h5netcdfMpi = (pyFor.h5netcdf.override {
                h5py = h5pyMpi;
              }).overridePythonAttrs (
                old: {
                  doCheck = false;
                  nativeCheckInputs = [ ];
                }
              );
              meshioMpi = (pyFor.meshio.override {
                h5py = h5pyMpi;
              }).overridePythonAttrs (
                old: {
                  doCheck = false;
                  nativeCheckInputs = [ ];
                }
              );
              arvizRuntime = (pyFor.arviz.override {
                h5netcdf = h5netcdfMpi;
              }).overridePythonAttrs (
                old: {
                  doCheck = false;
                  nativeCheckInputs = [ ];
                }
              );
              versioneer026 = pyFor.buildPythonPackage rec {
                pname = "versioneer";
                version = "0.26";
                pyproject = true;

                src = pkgsFor.fetchPypi {
                  inherit pname version;
                  hash = "sha256-hPxymqKW0dJmRaj2LxeAGYhf9vmhBzsppKIoJwrFJXs=";
                };

                build-system = [ pyFor.setuptools ];
                dependencies = [ pyFor.tomli ];
                doCheck = false;
                pythonImportsCheck = [ "versioneer" ];
              };
              cuqipy = pyFor.buildPythonPackage rec {
                pname = "cuqipy";
                version = "1.5.0";
                pyproject = true;

                src = pkgsFor.fetchPypi {
                  inherit pname version;
                  hash = "sha256-EBl7lX4kyWqRSkUUiGJFy5xxjIZm2JhQ/Jme9xS7GGQ=";
                };

                build-system = [
                  pyFor.setuptools
                  versioneer026
                ];

                dependencies = [
                  arvizRuntime
                  pyFor.matplotlib
                  pyFor.numpy
                  pyFor.scipy
                  pyFor.tqdm
                ];

                pythonRelaxDeps = [ "numpy" ];
                doCheck = false;
                pythonImportsCheck = [ "cuqi" ];
              };
              pyvistaqt = pyFor.buildPythonPackage rec {
                pname = "pyvistaqt";
                version = "0.11.4";
                pyproject = true;

                src = pkgsFor.fetchPypi {
                  inherit pname version;
                  hash = "sha256-srySrDTiu9cpxV+ydxloTUtRilk/zpyHdiECDQULS9U=";
                };

                build-system = [
                  pyFor.setuptools
                  pyFor.setuptools-scm
                ];

                dependencies = [
                  pyFor.pyvista
                  pyFor.qtpy
                ];

                doCheck = false;
                pythonImportsCheck = [ "pyvistaqt" ];
              };
            in
            {
              inherit h5pyMpi meshioMpi cuqipy pyvistaqt;
            };
          backendWorkerCommandEnvName = profile:
            "EIT_APP_BACKEND_WORKER_COMMAND_${lib.toUpper (lib.replaceStrings [ "-" "." ] [ "_" "_" ] profile)}";

          mkPyeidors = {
            pkgsFor,
            pyFor,
            profile,
            fenicsDolfinxPkg,
            petsc4pyPkg,
            cuda ? false,
            petscPkg ? null,
            slepcPkg ? null,
            scalarEnv ? null,
            precisionEnv ? null,
            backendWorkerCommands ? { },
          }:
            let
              runtime = mkRuntimePython { inherit pkgsFor pyFor; };
              profileSuffix = if profile == "default" then "" else "-${profile}";
              packageName = if profile == "default" then "pyeidors" else "pyeidors${profileSuffix}";
              cudaRuntimeLibs = lib.optionals cuda [
                pkgsFor.cudaPackages.cuda_cudart
                pkgsFor.cudaPackages.libcublas
                pkgsFor.cudaPackages.libcusolver
                pkgsFor.cudaPackages.libcusparse
                pkgsFor.cudaPackages.libnvjitlink
              ];
            in
            pyFor.buildPythonApplication {
              pname = packageName;
              version = pyeidorsVersion;
              pyproject = true;

              src = pyeidorsSource;

              build-system = [
                pyFor.setuptools
                pyFor.wheel
              ];

              dependencies = [
                fenicsDolfinxPkg
                pyFor."fenics-basix"
                pyFor."fenics-ffcx"
                pyFor."fenics-ufl"
                pyFor.mpi4py
                petsc4pyPkg

                pyFor.numpy
                pyFor.scipy
                pyFor.matplotlib
                pyFor.pandas
                runtime.h5pyMpi
                pyFor.pyyaml
                runtime.meshioMpi
                pyFor.gmsh

                pyFor.pyside6
                pyFor.pyqtgraph
                pyFor.pyserial
                pyFor.pyvista
                runtime.pyvistaqt

                pyFor.torch
                runtime.cuqipy
              ] ++ pyOpt pyFor "pyamg" ++ pyOpt pyFor "scikit-sparse" ++ pyOpt pyFor "scikitsparse";

              doCheck = false;
              pythonImportsCheck = [
                "dolfinx"
                "pyeidors"
                "eit_app"
                "eit_app.ecd_cwr_simulation"
                "PySide6.QtCore"
                "pyqtgraph"
                "pyvista"
                "pyvistaqt"
                "torch"
                "cuqi"
              ];

              makeWrapperArgs = [
                "--unset"
                "PYTHONHOME"
                "--unset"
                "PYTHONPATH"
                "--unset"
                "PYTHONSTARTUP"
                "--unset"
                "PYTHONUSERBASE"
                "--unset"
                "PYTHONWARNINGS"
                "--unset"
                "VIRTUAL_ENV"
                "--unset"
                "CONDA_PREFIX"
                "--unset"
                "CONDA_DEFAULT_ENV"
                "--unset"
                "CONDA_SHLVL"
                "--unset"
                "LD_PRELOAD"
                "--unset"
                "CMAKE_PREFIX_PATH"
                "--set"
                "PYTHONNOUSERSITE"
                "1"
                "--set-default"
                "PYEIDORS_ENV_PROFILE"
                profile
                "--set-default"
                "EIT_APP_GUI_RUNTIME_PROFILE"
                profile
                "--set-default"
                "EIT_APP_GUI_PROFILE"
                (if cuda then "gpu" else "cpu")
                "--set-default"
                "EIT_APP_BACKEND_WORKER_LAUNCH_MODE"
                "auto"
                "--set-default"
                "EIT_APP_3D_WSLG_PYVISTA_OFFSCREEN"
                "1"
                "--set"
                "CC"
                "${pkgsFor.stdenv.cc}/bin/cc"
                "--set"
                "CXX"
                "${pkgsFor.stdenv.cc}/bin/c++"
                "--prefix"
                "PATH"
                ":"
                (lib.makeBinPath (
                  [
                    pkgsFor.coreutils
                    pkgsFor.stdenv.cc
                    pkgsFor.gmsh
                    pkgsFor.openmpi
                    pkgsFor.fontconfig.bin
                  ] ++ lib.optionals cuda [
                    pkgsFor.cudaPackages.cuda_nvcc
                    pkgsFor.cudaPackages.cudatoolkit
                  ]
                ))
              ] ++ lib.concatLists (lib.mapAttrsToList
                (workerProfile: workerPackage: [
                  "--set-default"
                  (backendWorkerCommandEnvName workerProfile)
                  "${workerPackage}/bin/eit-backend-worker"
                ] ++ lib.optionals (workerProfile == "cuda-amgx") [
                  "--set-default"
                  "PYEIDORS_BLOCK_REAL_AMGX_WORKER_COMMAND"
                  "${workerPackage}/bin/pyeidors-block-real-amgx"
                ])
                backendWorkerCommands) ++ lib.optionals (scalarEnv != null) [
                "--set-default"
                "PYEIDORS_PETSC_SCALAR_TYPE"
                scalarEnv
              ] ++ lib.optionals (precisionEnv != null) [
                "--set-default"
                "EIT_APP_GUI_PRECISION"
                precisionEnv
              ] ++ lib.optionals cuda [
                "--set"
                "CUDA_HOME"
                "${pkgsFor.cudaPackages.cudatoolkit}"
                "--set"
                "CUDA_PATH"
                "${pkgsFor.cudaPackages.cudatoolkit}"
                "--set"
                "CUDACXX"
                "${pkgsFor.cudaPackages.cuda_nvcc}/bin/nvcc"
                "--set"
                "PETSC_DIR"
                "${petscPkg}"
                "--set"
                "SLEPC_DIR"
                "${slepcPkg}"
                "--set-default"
                "PYEIDORS_PETSC_DEVICE_DEFAULT"
                "cuda"
                ''--run 'export PETSC_OPTIONS="-use_gpu_aware_mpi 0 -nox_warning''${PETSC_OPTIONS:+ $PETSC_OPTIONS}"' ''
                "--prefix"
                "PATH"
                ":"
                "/usr/lib/wsl/lib"
              ] ++ lib.optionals pkgsFor.stdenv.isLinux [
                "--prefix"
                "LD_LIBRARY_PATH"
                ":"
                (lib.makeLibraryPath ([ pkgsFor.stdenv.cc.cc pkgsFor.zlib pkgsFor.zstd ] ++ mkLinuxGuiLibs pkgsFor ++ cudaRuntimeLibs))
                "--set"
                "LIBGL_DRIVERS_PATH"
                "${pkgsFor.mesa}/lib/dri"
              ] ++ lib.optionals (pkgsFor.stdenv.isLinux && cuda) [
                "--prefix"
                "LD_LIBRARY_PATH"
                ":"
                "/usr/lib/wsl/lib"
              ] ++ lib.optionals pkgsFor.stdenv.isLinux [
                ''--run 'export PYEIDORS_RUNTIME_ROOT="''${PYEIDORS_RUNTIME_ROOT:-''${XDG_CACHE_HOME:-$HOME/.cache}/pyeidors}"' ''
                ''--run 'export PYEIDORS_CACHE_ROOT="''${PYEIDORS_CACHE_ROOT:-$PYEIDORS_RUNTIME_ROOT/.pyeidors_cache}"' ''
                ''--run 'export PYEIDORS_DATA_ROOT="''${PYEIDORS_DATA_ROOT:-''${XDG_DATA_HOME:-$HOME/.local/share}/pyeidors}"' ''
                ''--run 'export PYEIDORS_OUTPUT_ROOT="''${PYEIDORS_OUTPUT_ROOT:-$PYEIDORS_DATA_ROOT/outputs}"' ''
                ''--run 'export PYEIDORS_GREIT_ARTIFACT_REGISTRY_DIR="''${PYEIDORS_GREIT_ARTIFACT_REGISTRY_DIR:-$PYEIDORS_CACHE_ROOT/greit_artifacts}"' ''
                ''--run 'export PYEIDORS_GREIT_COMMON_CONFIG_DIR="''${PYEIDORS_GREIT_COMMON_CONFIG_DIR:-$PYEIDORS_CACHE_ROOT/greit_common_configs}"' ''
                ''--run 'export EIT_APP_BACKEND_WORKER_CACHE_DIR="''${EIT_APP_BACKEND_WORKER_CACHE_DIR:-$PYEIDORS_CACHE_ROOT/gui_backend_worker}"' ''
                ''--run 'mkdir -p "$PYEIDORS_RUNTIME_ROOT" "$PYEIDORS_CACHE_ROOT" "$PYEIDORS_DATA_ROOT" "$PYEIDORS_OUTPUT_ROOT" "$PYEIDORS_GREIT_ARTIFACT_REGISTRY_DIR" "$PYEIDORS_GREIT_COMMON_CONFIG_DIR" "$EIT_APP_BACKEND_WORKER_CACHE_DIR"' ''
              ];

              meta = with lib; {
                description = "PyEIDORS ${profile} FEniCSx runtime and GUI packaged as a Nix application";
                homepage = "https://github.com/CBZ199671/PyEIDORS";
                license = licenses.mit;
                mainProgram = "eit-app";
                platforms = platforms.linux ++ platforms.darwin;
              };
            };

          pyeidors = mkPyeidors {
            pkgsFor = pkgs;
            pyFor = py;
            profile = "default";
            fenicsDolfinxPkg = fenicsDolfinx;
            petsc4pyPkg = py.petsc4py;
          };
          pyeidorsComplex = mkPyeidors {
            pkgsFor = pkgs;
            pyFor = py;
            profile = "complex";
            fenicsDolfinxPkg = fenicsDolfinxComplex;
            petsc4pyPkg = petsc4pyComplex;
            scalarEnv = "complex";
            precisionEnv = "complex128";
            backendWorkerCommands = {
              default = pyeidors;
            };
          };
          pyeidorsComplex64 = mkPyeidors {
            pkgsFor = pkgs;
            pyFor = py;
            profile = "complex64";
            fenicsDolfinxPkg = fenicsDolfinxComplexSingle;
            petsc4pyPkg = petsc4pyComplexSingle;
            scalarEnv = "complex64";
            precisionEnv = "complex64";
            backendWorkerCommands = {
              default = pyeidors;
            };
          };
          pyeidorsCuda = if linuxCudaSupported then mkPyeidors {
            pkgsFor = pkgsCuda;
            pyFor = pyCuda;
            profile = "cuda";
            fenicsDolfinxPkg = cudaFenicsDolfinx;
            petsc4pyPkg = cudaPetsc4py;
            cuda = true;
            petscPkg = cudaPetsc;
            slepcPkg = cudaSlepc;
            backendWorkerCommands = {
              default = pyeidors;
            };
          } else null;
          pyeidorsCudaAmgx = if linuxCudaSupported then mkPyeidors {
            pkgsFor = pkgsCuda;
            pyFor = pyCuda;
            profile = "cuda-amgx";
            fenicsDolfinxPkg = cudaFenicsDolfinxAmgx;
            petsc4pyPkg = cudaPetscAmgx4py;
            cuda = true;
            petscPkg = cudaPetscAmgx;
            slepcPkg = cudaSlepcAmgx;
            backendWorkerCommands = {
              default = pyeidors;
              cuda = pyeidorsCuda;
            };
          } else null;
          pyeidorsComplexCuda = if linuxCudaSupported then mkPyeidors {
            pkgsFor = pkgsCuda;
            pyFor = pyCuda;
            profile = "complex-cuda";
            fenicsDolfinxPkg = cudaFenicsDolfinxComplex;
            petsc4pyPkg = cudaPetscComplex4py;
            cuda = true;
            petscPkg = cudaPetscComplex;
            slepcPkg = cudaSlepcComplex;
            scalarEnv = "complex";
            precisionEnv = "complex128";
            backendWorkerCommands = {
              default = pyeidors;
              cuda = pyeidorsCuda;
              cuda-amgx = pyeidorsCudaAmgx;
            };
          } else null;
          pyeidorsComplexCudaAmgx = if linuxCudaSupported then mkPyeidors {
            pkgsFor = pkgsCuda;
            pyFor = pyCuda;
            profile = "complex-cuda-amgx";
            fenicsDolfinxPkg = cudaFenicsDolfinxComplexAmgx;
            petsc4pyPkg = cudaPetscComplexAmgx4py;
            cuda = true;
            petscPkg = cudaPetscComplexAmgx;
            slepcPkg = cudaSlepcComplexAmgx;
            scalarEnv = "complex";
            precisionEnv = "complex128";
            backendWorkerCommands = {
              default = pyeidors;
              cuda = pyeidorsCudaAmgx;
              cuda-amgx = pyeidorsCudaAmgx;
            };
          } else null;
          pyeidorsComplex64Cuda = if linuxCudaSupported then mkPyeidors {
            pkgsFor = pkgsCuda;
            pyFor = pyCuda;
            profile = "complex64-cuda";
            fenicsDolfinxPkg = cudaFenicsDolfinxComplexSingle;
            petsc4pyPkg = cudaPetscComplexSingle4py;
            cuda = true;
            petscPkg = cudaPetscComplexSingle;
            slepcPkg = cudaSlepcComplexSingle;
            scalarEnv = "complex64";
            precisionEnv = "complex64";
            backendWorkerCommands = {
              default = pyeidors;
              cuda = pyeidorsCuda;
              cuda-amgx = pyeidorsCudaAmgx;
            };
          } else null;
          pyeidorsCudaSm61 = if linuxCudaSupported then mkPyeidors {
            pkgsFor = pkgsCudaSm61;
            pyFor = pyCudaSm61;
            profile = "cuda-sm61";
            fenicsDolfinxPkg = cudaSm61FenicsDolfinx;
            petsc4pyPkg = cudaSm61Petsc4py;
            cuda = true;
            petscPkg = cudaSm61Petsc;
            slepcPkg = cudaSm61Slepc;
            backendWorkerCommands = {
              default = pyeidors;
            };
          } else null;
          pyeidorsComplex64CudaSm61 = if linuxCudaSupported then mkPyeidors {
            pkgsFor = pkgsCudaSm61;
            pyFor = pyCudaSm61;
            profile = "complex64-cuda-sm61";
            fenicsDolfinxPkg = cudaSm61FenicsDolfinxComplexSingle;
            petsc4pyPkg = cudaSm61PetscComplexSingle4py;
            cuda = true;
            petscPkg = cudaSm61PetscComplexSingle;
            slepcPkg = cudaSm61SlepcComplexSingle;
            scalarEnv = "complex64";
            precisionEnv = "complex64";
            backendWorkerCommands = {
              default = pyeidors;
              cuda-sm61 = pyeidorsCudaSm61;
            };
          } else null;
        in
        {
          inherit pyeidors;
          pyeidors-default = pyeidors;
          pyeidors-complex = pyeidorsComplex;
          pyeidors-complex64 = pyeidorsComplex64;
          default = pyeidors;
        } // lib.optionalAttrs linuxCudaSupported {
          pyeidors-cuda = pyeidorsCuda;
          pyeidors-cuda-amgx = pyeidorsCudaAmgx;
          pyeidors-complex-cuda = pyeidorsComplexCuda;
          pyeidors-complex-cuda-amgx = pyeidorsComplexCudaAmgx;
          pyeidors-complex64-cuda = pyeidorsComplex64Cuda;
          pyeidors-cuda-sm61 = pyeidorsCudaSm61;
          pyeidors-complex64-cuda-sm61 = pyeidorsComplex64CudaSm61;
        }
      );

      apps = forAllSystems (
        system:
        let
          packagesForSystem = self.packages.${system};
          mkApp = packageName: programName: description: {
            type = "app";
            program = "${builtins.getAttr packageName packagesForSystem}/bin/${programName}";
            meta.description = description;
          };
          hasPackage = name: builtins.hasAttr name packagesForSystem;
        in
        {
          default = mkApp "pyeidors" "eit-app" "Launch the PyEIDORS default real-valued CPU GUI";
          eit-app = mkApp "pyeidors" "eit-app" "Launch the PyEIDORS default real-valued CPU GUI";
          eit-cache = mkApp "pyeidors" "eit-cache" "Manage and warm PyEIDORS default real-valued CPU caches";
          eit-app-default = mkApp "pyeidors" "eit-app" "Launch the PyEIDORS default real-valued CPU GUI";
          eit-cache-default = mkApp "pyeidors" "eit-cache" "Manage and warm PyEIDORS default real-valued CPU caches";
          eit-app-real-cpu = mkApp "pyeidors" "eit-app" "Launch the PyEIDORS real-valued CPU GUI";
          eit-cache-real-cpu = mkApp "pyeidors" "eit-cache" "Manage and warm PyEIDORS real-valued CPU caches";
          eit-backend-worker-real-cpu = mkApp "pyeidors" "eit-backend-worker" "Run the PyEIDORS real-valued CPU backend worker";
          eit-backend-doctor-real-cpu = mkApp "pyeidors" "eit-backend-doctor" "Check the PyEIDORS real-valued CPU backend runtime";
          eit-app-complex = mkApp "pyeidors-complex" "eit-app" "Launch the PyEIDORS complex128 CPU GUI";
          eit-cache-complex = mkApp "pyeidors-complex" "eit-cache" "Manage and warm PyEIDORS complex128 CPU caches";
          eit-app-complex128-cpu = mkApp "pyeidors-complex" "eit-app" "Launch the PyEIDORS complex128 CPU GUI";
          eit-cache-complex128-cpu = mkApp "pyeidors-complex" "eit-cache" "Manage and warm PyEIDORS complex128 CPU caches";
          eit-backend-worker-complex128-cpu = mkApp "pyeidors-complex" "eit-backend-worker" "Run the PyEIDORS complex128 CPU backend worker";
          eit-backend-doctor-complex128-cpu = mkApp "pyeidors-complex" "eit-backend-doctor" "Check the PyEIDORS complex128 CPU backend runtime";
          eit-app-complex64 = mkApp "pyeidors-complex64" "eit-app" "Launch the PyEIDORS complex64 CPU GUI";
          eit-cache-complex64 = mkApp "pyeidors-complex64" "eit-cache" "Manage and warm PyEIDORS complex64 CPU caches";
          eit-app-complex64-cpu = mkApp "pyeidors-complex64" "eit-app" "Launch the PyEIDORS complex64 CPU GUI";
          eit-cache-complex64-cpu = mkApp "pyeidors-complex64" "eit-cache" "Manage and warm PyEIDORS complex64 CPU caches";
          eit-backend-worker-complex64 = mkApp "pyeidors-complex64" "eit-backend-worker" "Run the PyEIDORS complex64 CPU backend worker";
          eit-backend-doctor-complex64 = mkApp "pyeidors-complex64" "eit-backend-doctor" "Check the PyEIDORS complex64 CPU backend runtime";
          eit-backend-worker-complex64-cpu = mkApp "pyeidors-complex64" "eit-backend-worker" "Run the PyEIDORS complex64 CPU backend worker";
          eit-backend-doctor-complex64-cpu = mkApp "pyeidors-complex64" "eit-backend-doctor" "Check the PyEIDORS complex64 CPU backend runtime";
        } // lib.optionalAttrs (hasPackage "pyeidors-cuda") {
          eit-app-cuda = mkApp "pyeidors-cuda" "eit-app" "Launch the PyEIDORS real-valued CUDA GUI";
          eit-cache-cuda = mkApp "pyeidors-cuda" "eit-cache" "Manage and warm PyEIDORS real-valued CUDA caches";
          eit-app-real-gpu = mkApp "pyeidors-cuda" "eit-app" "Launch the PyEIDORS real-valued CUDA GUI";
          eit-cache-real-gpu = mkApp "pyeidors-cuda" "eit-cache" "Manage and warm PyEIDORS real-valued CUDA caches";
          eit-backend-worker-real-gpu = mkApp "pyeidors-cuda" "eit-backend-worker" "Run the PyEIDORS real-valued CUDA backend worker";
          eit-backend-doctor-real-gpu = mkApp "pyeidors-cuda" "eit-backend-doctor" "Check the PyEIDORS real-valued CUDA backend runtime";
          eit-app-cuda-amgx = mkApp "pyeidors-cuda-amgx" "eit-app" "Launch the PyEIDORS real-valued CUDA AmgX GUI";
          eit-cache-cuda-amgx = mkApp "pyeidors-cuda-amgx" "eit-cache" "Manage and warm PyEIDORS real-valued CUDA AmgX caches";
          eit-backend-worker-cuda-amgx = mkApp "pyeidors-cuda-amgx" "eit-backend-worker" "Run the PyEIDORS CUDA AmgX backend worker";
          eit-backend-doctor-cuda-amgx = mkApp "pyeidors-cuda-amgx" "eit-backend-doctor" "Check the PyEIDORS CUDA AmgX backend runtime";
          eit-app-complex-cuda = mkApp "pyeidors-complex-cuda" "eit-app" "Launch the PyEIDORS complex128 CUDA GUI";
          eit-cache-complex-cuda = mkApp "pyeidors-complex-cuda" "eit-cache" "Manage and warm PyEIDORS complex128 CUDA caches";
          eit-app-complex128-gpu = mkApp "pyeidors-complex-cuda" "eit-app" "Launch the PyEIDORS complex128 CUDA GUI";
          eit-cache-complex128-gpu = mkApp "pyeidors-complex-cuda" "eit-cache" "Manage and warm PyEIDORS complex128 CUDA caches";
          eit-backend-worker-complex128-gpu = mkApp "pyeidors-complex-cuda" "eit-backend-worker" "Run the PyEIDORS complex128 CUDA backend worker";
          eit-backend-doctor-complex128-gpu = mkApp "pyeidors-complex-cuda" "eit-backend-doctor" "Check the PyEIDORS complex128 CUDA backend runtime";
          eit-app-complex-cuda-amgx = mkApp "pyeidors-complex-cuda-amgx" "eit-app" "Launch the PyEIDORS experimental complex128 CUDA AmgX GUI";
          eit-cache-complex-cuda-amgx = mkApp "pyeidors-complex-cuda-amgx" "eit-cache" "Manage and warm PyEIDORS experimental complex128 CUDA AmgX caches";
          eit-backend-worker-complex128-gpu-amgx = mkApp "pyeidors-complex-cuda-amgx" "eit-backend-worker" "Run the PyEIDORS experimental complex128 CUDA AmgX backend worker";
          eit-backend-doctor-complex128-gpu-amgx = mkApp "pyeidors-complex-cuda-amgx" "eit-backend-doctor" "Check the PyEIDORS experimental complex128 CUDA AmgX backend runtime";
          eit-app-complex64-cuda = mkApp "pyeidors-complex64-cuda" "eit-app" "Launch the PyEIDORS complex64 CUDA GUI";
          eit-cache-complex64-cuda = mkApp "pyeidors-complex64-cuda" "eit-cache" "Manage and warm PyEIDORS complex64 CUDA caches";
          eit-app-complex64-gpu = mkApp "pyeidors-complex64-cuda" "eit-app" "Launch the PyEIDORS complex64 CUDA GUI";
          eit-cache-complex64-gpu = mkApp "pyeidors-complex64-cuda" "eit-cache" "Manage and warm PyEIDORS complex64 CUDA caches";
          eit-backend-worker-complex64-cuda = mkApp "pyeidors-complex64-cuda" "eit-backend-worker" "Run the PyEIDORS complex64 CUDA backend worker";
          eit-backend-doctor-complex64-cuda = mkApp "pyeidors-complex64-cuda" "eit-backend-doctor" "Check the PyEIDORS complex64 CUDA backend runtime";
          eit-app-cuda-sm61 = mkApp "pyeidors-cuda-sm61" "eit-app" "Launch the PyEIDORS real-valued CUDA GUI for sm_61 Pascal GPUs";
          eit-cache-cuda-sm61 = mkApp "pyeidors-cuda-sm61" "eit-cache" "Manage and warm PyEIDORS CUDA caches for sm_61 Pascal GPUs";
          eit-backend-worker-real-gpu-sm61 = mkApp "pyeidors-cuda-sm61" "eit-backend-worker" "Run the PyEIDORS real-valued CUDA backend worker for sm_61 Pascal GPUs";
          eit-backend-doctor-real-gpu-sm61 = mkApp "pyeidors-cuda-sm61" "eit-backend-doctor" "Check the PyEIDORS real-valued CUDA backend runtime for sm_61 Pascal GPUs";
          eit-app-complex64-cuda-sm61 = mkApp "pyeidors-complex64-cuda-sm61" "eit-app" "Launch the PyEIDORS complex64 CUDA GUI for sm_61 Pascal GPUs";
          eit-cache-complex64-cuda-sm61 = mkApp "pyeidors-complex64-cuda-sm61" "eit-cache" "Manage and warm PyEIDORS complex64 CUDA caches for sm_61 Pascal GPUs";
          eit-backend-worker-complex64-cuda-sm61 = mkApp "pyeidors-complex64-cuda-sm61" "eit-backend-worker" "Run the PyEIDORS complex64 CUDA backend worker for sm_61 Pascal GPUs";
          eit-backend-doctor-complex64-cuda-sm61 = mkApp "pyeidors-complex64-cuda-sm61" "eit-backend-doctor" "Check the PyEIDORS complex64 CUDA backend runtime for sm_61 Pascal GPUs";
          eit-app-legacy-gpu = mkApp "pyeidors-complex64-cuda-sm61" "eit-app" "Launch the PyEIDORS complex64 CUDA GUI for sm_61 Pascal GPUs";
          eit-cache-legacy-gpu = mkApp "pyeidors-complex64-cuda-sm61" "eit-cache" "Manage and warm PyEIDORS complex64 CUDA caches for sm_61 Pascal GPUs";
        }
      );

      devShells = forAllSystems (
        system:
        let
          nixpkgsPath = nixpkgs.outPath;
          pkgs = import nixpkgs { inherit system; };
          pyeidorsPackages = self.packages.${system};
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
          amgxSource = if linuxCudaSupported then pkgsCuda.fetchgit {
            url = amgxGitUrl;
            rev = amgxGitCommit;
            hash = amgxSourceHash;
            fetchSubmodules = true;
          } else null;
          amgxSourceArchive = if linuxCudaSupported then pkgsCuda.runCommand
            "AMGX-${lib.substring 0 7 amgxGitCommit}.tar.gz"
            { nativeBuildInputs = [ pkgsCuda.gnutar pkgsCuda.gzip ]; }
            ''
              cp -R --no-preserve=mode,ownership ${amgxSource} "$TMPDIR/AMGX-${amgxGitCommit}"
              chmod -R u+rwX,go+rX "$TMPDIR/AMGX-${amgxGitCommit}"
              cp ${pkgsCuda.cudaPackages.cuda_nvtx.include}/include/nvToolsExt.h \
                "$TMPDIR/AMGX-${amgxGitCommit}/include/nvToolsExt.h"
              cp ${pkgsCuda.cudaPackages.libcurand.include}/include/curand*.h \
                "$TMPDIR/AMGX-${amgxGitCommit}/include/"
              substituteInPlace "$TMPDIR/AMGX-${amgxGitCommit}/CMakeLists.txt" \
                --replace-fail 'target_link_libraries(amgx CUDA::cublas CUDA::cusparse CUDA::cusolver CUDA::nvToolsExt m pthread)' \
                  'target_link_libraries(amgx CUDA::cublas CUDA::cusparse CUDA::cusolver CUDA::nvToolsExt m pthread "${pkgsCuda.openmpi}/lib/libmpi.so")' \
                --replace-fail 'target_link_libraries(amgxsh CUDA::cublas CUDA::cusparse CUDA::cusolver CUDA::nvToolsExt m pthread)' \
                  'target_link_libraries(amgxsh CUDA::cublas CUDA::cusparse CUDA::cusolver CUDA::nvToolsExt m pthread "${pkgsCuda.openmpi}/lib/libmpi.so")'
              substituteInPlace "$TMPDIR/AMGX-${amgxGitCommit}/src/CMakeLists.txt" \
                --replace-fail 'target_link_libraries(amgx_tests_launcher amgxsh ''${libs_all} OpenMP::OpenMP_C)' \
                  'target_link_libraries(amgx_tests_launcher amgxsh ''${libs_all} OpenMP::OpenMP_C "${pkgsCuda.openmpi}/lib/libmpi.so")'
              tar -C "$TMPDIR" \
                --owner=0 \
                --group=0 \
                --numeric-owner \
                -czf "$out" AMGX-${amgxGitCommit}
            ''
          else null;
          cmakeForPetscExternalPackages = if linuxCudaSupported then pkgsCuda.runCommand "cmake-for-petsc-external-packages" { } ''
            mkdir -p "$out/bin"
            for tool in cmake ctest cpack; do
              ln -s "${pkgsCuda.cmake}/bin/$tool" "$out/bin/$tool"
            done
          '' else null;
          mkCudaPetsc = { scalarType ? null, precision ? null, withAmgx ? false, allowComplexAmgx ? false }:
            assert !withAmgx || allowComplexAmgx || scalarType == null || scalarType == "real";
            assert !withAmgx || allowComplexAmgx || precision == null || precision == "double";
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
                postPatch = (old.postPatch or "") + lib.optionalString (withAmgx && allowComplexAmgx) ''
                  substituteInPlace src/ksp/pc/impls/amgx/amgx.cxx \
                    --replace-fail '  PetscScalar      strength_threshold;' '  PetscReal        strength_threshold;' \
                    --replace-fail '  PetscScalar jacobi_relaxation_factor;' '  PetscReal jacobi_relaxation_factor;' \
                    --replace-fail '  PetscScalar gs_symmetric;' '  PetscReal gs_symmetric;' \
                    --replace-fail '    PetscCall(PetscOptionsScalar("-pc_amgx_jacobi_relaxation_factor", "AmgX AMG Jacobi Relaxation Factor", "", amgx->jacobi_relaxation_factor, &amgx->jacobi_relaxation_factor, NULL));' '    PetscCall(PetscOptionsReal("-pc_amgx_jacobi_relaxation_factor", "AmgX AMG Jacobi Relaxation Factor", "", amgx->jacobi_relaxation_factor, &amgx->jacobi_relaxation_factor, NULL));' \
                    --replace-fail '    PetscCall(PetscOptionsScalar("-pc_amgx_gs_symmetric", "AmgX AMG Gauss Seidel Symmetric", "", amgx->gs_symmetric, &amgx->gs_symmetric, NULL));' '    PetscCall(PetscOptionsReal("-pc_amgx_gs_symmetric", "AmgX AMG Gauss Seidel Symmetric", "", amgx->gs_symmetric, &amgx->gs_symmetric, NULL));' \
                    --replace-fail '  PetscCall(PetscOptionsScalar("-pc_amgx_strength_threshold", "AmgX AMG Strength Threshold", "", amgx->strength_threshold, &amgx->strength_threshold, NULL));' '  PetscCall(PetscOptionsReal("-pc_amgx_strength_threshold", "AmgX AMG Strength Threshold", "", amgx->strength_threshold, &amgx->strength_threshold, NULL));'
                  substituteInPlace src/ksp/pc/impls/amgx/amgx.cxx \
                    --replace-fail '    PetscCallAmgX(AMGX_matrix_create(&amgx->A, amgx->rsrc, AMGX_mode_dDDI));' '#if defined(PETSC_USE_COMPLEX) && defined(PETSC_USE_REAL_SINGLE)
                    const AMGX_Mode amgxMode = AMGX_mode_dCCI;
                  #elif defined(PETSC_USE_COMPLEX)
                    const AMGX_Mode amgxMode = AMGX_mode_dZZI;
                  #elif defined(PETSC_USE_REAL_SINGLE)
                    const AMGX_Mode amgxMode = AMGX_mode_dFFI;
                  #else
                    const AMGX_Mode amgxMode = AMGX_mode_dDDI;
                  #endif
                    PetscCallAmgX(AMGX_matrix_create(&amgx->A, amgx->rsrc, amgxMode));' \
                    --replace-fail '    PetscCallAmgX(AMGX_vector_create(&amgx->sol, amgx->rsrc, AMGX_mode_dDDI));' '    PetscCallAmgX(AMGX_vector_create(&amgx->sol, amgx->rsrc, amgxMode));' \
                    --replace-fail '    PetscCallAmgX(AMGX_vector_create(&amgx->rhs, amgx->rsrc, AMGX_mode_dDDI));' '    PetscCallAmgX(AMGX_vector_create(&amgx->rhs, amgx->rsrc, amgxMode));' \
                    --replace-fail '    PetscCallAmgX(AMGX_solver_create(&amgx->solver, amgx->rsrc, AMGX_mode_dDDI, amgx->cfg));' '    PetscCallAmgX(AMGX_solver_create(&amgx->solver, amgx->rsrc, amgxMode, amgx->cfg));'
                  substituteInPlace src/ksp/pc/impls/amgx/amgx.cxx \
                    --replace-fail '  PetscCheck(AmgXControlMap::Selectors.count(option) == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Selector %s not registered for AmgX.", option);' '  PetscCheck(AmgXControlMap::Selectors.count(option) == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Selector %s not registered for AmgX.", option);
                  amgx->selector = AmgXControlMap::Selectors.at(option);'
                '';
                nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                  pkgsCuda.cudaPackages.cuda_nvcc
                ] ++ lib.optional withAmgx cmakeForPetscExternalPackages;
                buildInputs = (old.buildInputs or [ ]) ++ [
                  pkgsCuda.cudaPackages.cuda_cudart
                  pkgsCuda.cudaPackages.libcublas
                  pkgsCuda.cudaPackages.libcusolver
                  pkgsCuda.cudaPackages.libcusparse
                ] ++ lib.optionals withAmgx [
                  pkgsCuda.cudaPackages.cuda_nvtx
                  pkgsCuda.cudaPackages.libcurand
                ];
                configureFlags = (old.configureFlags or [ ]) ++ [
                  "--with-cuda=1"
                  "--with-cudac=${pkgsCuda.cudaPackages.cuda_nvcc}/bin/nvcc"
                  "--with-cuda-dir=${pkgsCuda.cudaPackages.cudatoolkit}"
                  "--with-cublas=1"
                  "--with-cusparse=1"
                  "--with-cusolver=1"
                ] ++ lib.optionals withAmgx [
                  "--download-amgx=${amgxSourceArchive}"
                  "--with-64-bit-indices=0"
                  "--with-cxx-dialect=17"
                  "--with-cuda-dialect=17"
                ];
                doInstallCheck = false;
                postInstall = lib.replaceStrings [ "--replace-fail" ] [ "--replace" ] (old.postInstall or "");
              })
            else null;
          cudaPetsc = mkCudaPetsc { };
          cudaPetscAmgx = mkCudaPetsc { withAmgx = true; };
          cudaPetscComplex = mkCudaPetsc { scalarType = "complex"; };
          cudaPetscComplexAmgx = mkCudaPetsc {
            scalarType = "complex";
            withAmgx = true;
            allowComplexAmgx = true;
          };
          cudaPetscComplexSingle = mkCudaPetsc { scalarType = "complex"; precision = "single"; };
          cudaPetsc4py = if linuxCudaSupported then pyCuda.toPythonModule cudaPetsc else null;
          cudaPetscAmgx4py = if linuxCudaSupported then pyCuda.toPythonModule cudaPetscAmgx else null;
          cudaPetscComplex4py = if linuxCudaSupported then pyCuda.toPythonModule cudaPetscComplex else null;
          cudaPetscComplexAmgx4py = if linuxCudaSupported then pyCuda.toPythonModule cudaPetscComplexAmgx else null;
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
          cudaSlepcAmgx = mkCudaSlepc cudaPetscAmgx;
          cudaSlepcComplex = mkCudaSlepc cudaPetscComplex;
          cudaSlepcComplexAmgx = mkCudaSlepc cudaPetscComplexAmgx;
          cudaSlepcComplexSingle = mkCudaSlepc cudaPetscComplexSingle;
          cudaSlepc4py = if linuxCudaSupported then pyCuda.toPythonModule cudaSlepc else null;
          cudaSlepcAmgx4py = if linuxCudaSupported then pyCuda.toPythonModule cudaSlepcAmgx else null;
          cudaSlepcComplex4py = if linuxCudaSupported then pyCuda.toPythonModule cudaSlepcComplex else null;
          cudaSlepcComplexAmgx4py = if linuxCudaSupported then pyCuda.toPythonModule cudaSlepcComplexAmgx else null;
          cudaSlepcComplexSingle4py = if linuxCudaSupported then pyCuda.toPythonModule cudaSlepcComplexSingle else null;

          mkCudaDolfinx = petscPkg: slepcPkg:
            if linuxCudaSupported then pkgsCuda.callPackage "${nixpkgsPath}/pkgs/by-name/do/dolfinx/package.nix" {
              python3Packages = pyCuda;
              petsc = petscPkg;
              slepc = slepcPkg;
            } else null;
          cudaDolfinx = mkCudaDolfinx cudaPetsc cudaSlepc;
          cudaDolfinxAmgx = mkCudaDolfinx cudaPetscAmgx cudaSlepcAmgx;
          cudaDolfinxComplex = mkCudaDolfinx cudaPetscComplex cudaSlepcComplex;
          cudaDolfinxComplexAmgx = mkCudaDolfinx cudaPetscComplexAmgx cudaSlepcComplexAmgx;
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
          cudaFenicsDolfinxAmgx = mkCudaFenicsDolfinx cudaDolfinxAmgx cudaPetscAmgx4py cudaSlepcAmgx4py;
          cudaFenicsDolfinxComplex = mkCudaFenicsDolfinx cudaDolfinxComplex cudaPetscComplex4py cudaSlepcComplex4py;
          cudaFenicsDolfinxComplexAmgx = mkCudaFenicsDolfinx cudaDolfinxComplexAmgx cudaPetscComplexAmgx4py cudaSlepcComplexAmgx4py;
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
              export PYEIDORS_NIX_PYTHON="$UV_PYTHON"
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
                  echo "[nix] Darwin linker flags sanitized: removed ''${_darwin_linker_fix[2]} invalid -L entries."
                fi
              fi

              if [ "$(uname -s)" = "Linux" ]; then
                export LD_LIBRARY_PATH="${lib.makeLibraryPath ([ pkgsFor.stdenv.cc.cc pkgsFor.zlib pkgsFor.zstd ] ++ extraLinuxRuntimeLibs)}${extraLinuxLibraryPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              fi

              unset VIRTUAL_ENV
              unset VIRTUAL_ENV_PROMPT
              export PYEIDORS_ACTIVE_ENV="nix"
              if [ -d "$PWD/src" ]; then
                case ":''${PYTHONPATH:-}:" in
                  *":$PWD/src:"*) ;;
                  *)
                    export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"
                    ;;
                esac
              fi

              if [ -f scripts/env/cache_session.sh ]; then
                # shellcheck disable=SC1091
                source scripts/env/cache_session.sh
                pyeidors_cache_session_init ".pyeidors_cache/v2"
              fi

              if [ -z "''${PYEIDORS_SHELL_HOOK_READY:-}" ]; then
                if [ "''${PYEIDORS_ENABLE_UV_SYNC:-0}" = "1" ] && [ -x scripts/env/sync_locked_env.sh ]; then
                  echo "[nix+uv legacy] Checking locked Python environment profile (torch+cuqi+dev+eit-app)..."
                  if ! scripts/env/sync_locked_env.sh --check; then
                    if [ "''${PYEIDORS_GUI_LAUNCH:-0}" = "1" ]; then
                      echo "[nix+uv legacy] Refreshing locked Python environment profile..."
                    else
                      echo "[nix+uv legacy] Drift detected. Attempting automatic repair..."
                    fi
                    if ! scripts/env/sync_locked_env.sh --repair; then
                      echo "[nix+uv legacy] ERROR: environment repair failed."
                      echo "[nix+uv legacy] Manual repair command: scripts/env/sync_locked_env.sh --repair"
                      exit 1
                    fi
                  fi
                else
                  "$UV_PYTHON" - <<'PY'
import importlib
import sys
import warnings

warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message=r"pkg_resources is deprecated as an API",
    module=r"(pkg_resources(\..*)?|setuptools\._vendor\.pkg_resources(\..*)?|cuqi(\..*)?)",
)
warnings.filterwarnings(
    action="ignore",
    category=PendingDeprecationWarning,
    message=r"Importing from numpy\.matlib is deprecated",
    module=r"(numpy\.matlib(\..*)?|cuqi(\..*)?)",
)

required = ("dolfinx", "torch", "cuqi", "numpy", "scipy", "pyeidors", "pyqtgraph")
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")

try:
    from PySide6.QtCore import Qt  # noqa: F401
except Exception as exc:
    missing.append(f"PySide6.QtCore: {exc}")

if missing:
    print("[nix] ERROR: core dependency import check failed:", file=sys.stderr)
    for item in missing:
        print(f"  - {item}", file=sys.stderr)
    raise SystemExit(1)

print("[nix] Core dependency import checks passed: dolfinx, torch, cuqi, numpy, scipy, pyeidors, PySide6.QtCore, pyqtgraph")
PY
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
    f"[nix] Optional performance extras status: "
    f"pyamg={status['pyamg']}, sksparse={status['sksparse']}, cholmod={cholmod}"
    + " (missing extras do not block the core environment)"
)
PY
)"
                  echo "$perf_status"
                fi

                if [ "$PYEIDORS_ENV_PROFILE" = "cuda" ] || [ "$PYEIDORS_ENV_PROFILE" = "cuda-amgx" ] || [ "$PYEIDORS_ENV_PROFILE" = "complex-cuda" ] || [ "$PYEIDORS_ENV_PROFILE" = "complex64-cuda" ]; then
                  echo "[nix] CUDA profile ready. Verify PETSc CUDA backend with:"
                  echo "  python scripts/diagnostics/probe_petsc_cuda.py --require cuda --pretty"
                fi

                if [ "$PYEIDORS_ENV_PROFILE" = "cuda-amgx" ]; then
                  echo "[nix] AmgX PETSc profile ready. Verify PCAMGX with a setup/solve smoke:"
                  echo "  python - <<'PY'"
                  echo "from petsc4py import PETSc"
                  echo "opts = PETSc.Options(); opts['pc_amgx_smoother'] = 'JACOBI_L1'; opts['pc_amgx_exact_coarse_solve'] = '0'"
                  echo "A = PETSc.Mat().createAIJ([2, 2], nnz=2); A.setValues([0, 1], [0, 1], [[4.0, 1.0], [1.0, 3.0]]); A.assemblyBegin(); A.assemblyEnd()"
                  echo "b = PETSc.Vec().createSeq(2); b.setValues([0, 1], [1.0, 2.0]); b.assemblyBegin(); b.assemblyEnd(); x = b.duplicate()"
                  echo "ksp = PETSc.KSP().create(); ksp.setOperators(A); ksp.setType('cg'); ksp.getPC().setType('amgx'); ksp.setFromOptions(); ksp.solve(b, x)"
                  echo "print(ksp.getPC().getType(), ksp.getConvergedReason(), x.getArray())"
                  echo "ksp.destroy(); A.destroy(); b.destroy(); x.destroy()"
                  echo "PY"
                fi

                if [ "$PYEIDORS_ENV_PROFILE" = "complex" ] || [ "$PYEIDORS_ENV_PROFILE" = "complex64" ] || [ "$PYEIDORS_ENV_PROFILE" = "complex-cuda" ] || [ "$PYEIDORS_ENV_PROFILE" = "complex64-cuda" ]; then
                  echo "[nix] Complex PETSc profile ready. Verify scalar type with:"
                  echo "  python - <<'PY'"
                  echo "from petsc4py import PETSc; import numpy as np; print(np.dtype(PETSc.ScalarType))"
                  echo "PY"
                fi

                echo "[nix] Dev shell ready ($PYEIDORS_ENV_PROFILE)."
                echo "[nix] Pure Nix runtime active; uv sync is opt-in via PYEIDORS_ENABLE_UV_SYNC=1."
                export PYEIDORS_SHELL_HOOK_READY=1
              fi
            '';
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.uv
              pkgs.nodejs
              python
              pyeidorsPackages.pyeidors
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
              py.ipykernel
              py.jupyterlab
              py.pandas
              py.h5py
              py.pyyaml
              py.meshio
              py.gmsh
            ] ++ pyOpt "pyamg" ++ pyOpt "scikit-sparse" ++ pyOpt "scikitsparse" ++ [
              py.pytest
              py."pytest-cov"
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
              pkgs.nodejs
              python
              pyeidorsPackages.pyeidors-complex
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
              pkgsCuda.nodejs
              pythonCuda
              pyeidorsPackages.pyeidors-cuda
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

          "cuda-amgx" = pkgsCuda.mkShell {
            packages = [
              pkgsCuda.uv
              pkgsCuda.nodejs
              pythonCuda
              pyeidorsPackages.pyeidors-cuda-amgx
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

              cudaPetscAmgx
              cudaPetscAmgx4py
              cudaSlepcAmgx
              cudaSlepcAmgx4py
              cudaFenicsDolfinxAmgx
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
            ];

            shellHook = mkShellHook {
              pkgsFor = pkgsCuda;
              pythonFor = pythonCuda;
              envProfile = "cuda-amgx";
              venvDir = ".venv-cuda-amgx";
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
                export PETSC_DIR="${cudaPetscAmgx}"
                export SLEPC_DIR="${cudaSlepcAmgx}"
                export PYEIDORS_PETSC_DEVICE_DEFAULT="cuda"
                export PYEIDORS_PETSC_AMGX_ENABLED="1"
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
              pkgsCuda.nodejs
              pythonCuda
              pyeidorsPackages.pyeidors-complex-cuda
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

          "complex-cuda-amgx" = pkgsCuda.mkShell {
            packages = [
              pkgsCuda.uv
              pkgsCuda.nodejs
              pythonCuda
              pyeidorsPackages.pyeidors-complex-cuda-amgx
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

              cudaPetscComplexAmgx
              cudaPetscComplexAmgx4py
              cudaSlepcComplexAmgx
              cudaSlepcComplexAmgx4py
              cudaFenicsDolfinxComplexAmgx
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
            ];

            shellHook = mkShellHook {
              pkgsFor = pkgsCuda;
              pythonFor = pythonCuda;
              envProfile = "complex-cuda-amgx";
              venvDir = ".venv-complex-cuda-amgx";
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
                export PETSC_DIR="${cudaPetscComplexAmgx}"
                export SLEPC_DIR="${cudaSlepcComplexAmgx}"
                export PYEIDORS_PETSC_DEVICE_DEFAULT="cuda"
                export PYEIDORS_PETSC_AMGX_ENABLED="1"
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
              pkgsCuda.nodejs
              pythonCuda
              pyeidorsPackages.pyeidors-complex64-cuda
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
              pkgs.nodejs
              python
              pyeidorsPackages.pyeidors-complex64
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

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _flake_text() -> str:
    return (REPO_ROOT / "flake.nix").read_text(encoding="utf-8")


def _complex_petsc_override_block(text: str) -> str:
    marker = "} // lib.optionalAttrs (scalarType != null) {"
    end_marker = "} // lib.optionalAttrs (precision != null) {"
    assert marker in text
    after_marker = text.split(marker, 1)[1]
    assert end_marker in after_marker
    return after_marker.split(end_marker, 1)[0]


def test_v131_complex_cuda_petsc_profiles_disable_real_scalar_external_packages():
    text = _flake_text()

    assert 'cudaPetscComplex = mkCudaPetsc { scalarType = "complex"; };' in text
    assert (
        'cudaPetscComplexSingle = mkCudaPetsc { scalarType = "complex"; precision = "single"; };'
        in text
    )

    override_block = _complex_petsc_override_block(text)
    assert "inherit scalarType;" in override_block
    assert "withHypre = false;" in override_block
    assert "withSuperLuDist = false;" in override_block
    assert "withFftw = false;" in override_block
    assert "withSuitesparse = false;" in override_block


def test_v131_complex_cuda_shells_select_complex_dolfinx_petsc_profiles():
    text = _flake_text()

    assert '"complex-cuda" = pkgsCuda.mkShell {' in text
    complex_cuda_block = text.split('"complex-cuda" = pkgsCuda.mkShell {', 1)[1].split(
        "shellHook =", 1
    )[0]
    assert 'envProfile = "complex-cuda";' in text
    assert 'export PYEIDORS_PETSC_SCALAR_TYPE="complex"' in text
    assert 'export PYEIDORS_PETSC_DEVICE_DEFAULT="cuda"' in text
    assert "cudaPetscComplex" in text
    assert "cudaFenicsDolfinxComplex" in text
    assert "pkgsCuda.nodejs" in complex_cuda_block

    assert '"complex64-cuda" = pkgsCuda.mkShell {' in text
    complex64_cuda_block = text.split('"complex64-cuda" = pkgsCuda.mkShell {', 1)[
        1
    ].split("shellHook =", 1)[0]
    assert 'envProfile = "complex64-cuda";' in text
    assert 'export PYEIDORS_PETSC_SCALAR_TYPE="complex64"' in text
    assert "cudaPetscComplexSingle" in text
    assert "cudaFenicsDolfinxComplexSingle" in text
    assert "pkgsCuda.nodejs" in complex64_cuda_block


def test_v622_nix_apps_match_dev_launcher_wslg_pyvista_offscreen_default():
    flake_text = _flake_text()
    launcher_text = (REPO_ROOT / "scripts" / "gui" / "run_eit_app.sh").read_text(
        encoding="utf-8"
    )

    assert "EIT_APP_3D_WSLG_PYVISTA_OFFSCREEN:=1" in launcher_text
    assert '"EIT_APP_3D_WSLG_PYVISTA_OFFSCREEN"' in flake_text
    assert '"1"\n                "--prefix"\n                "PATH"' in flake_text


def test_v624_complex_nix_apps_expose_real_worker_profile_commands():
    text = _flake_text()

    assert '"EIT_APP_BACKEND_WORKER_LAUNCH_MODE"' in text
    assert '"auto"' in text
    assert "backendWorkerCommandEnvName" in text
    assert '"EIT_APP_BACKEND_WORKER_COMMAND_' in text
    assert '"${workerPackage}/bin/eit-backend-worker"' in text
    assert '"PYEIDORS_BLOCK_REAL_AMGX_WORKER_COMMAND"' in text
    assert '"${workerPackage}/bin/pyeidors-block-real-amgx"' in text
    assert "cuda = pyeidorsCuda;" in text
    assert "backendWorkerCommands = {\n              default = pyeidors;" in text
    assert (
        'profile = "cuda";\n            fenicsDolfinxPkg = cudaFenicsDolfinx;' in text
    )
    assert "pyeidorsCuda = if linuxCudaSupported then mkPyeidors" in text
    cuda_block = text.split(
        "pyeidorsCuda = if linuxCudaSupported then mkPyeidors",
        1,
    )[1].split("} else null;", 1)[0]
    assert "backendWorkerCommands = {\n              default = pyeidors;" in cuda_block


def test_cuda_amgx_profile_is_explicit_real_double_nix_route():
    text = _flake_text()

    assert 'amgxGitCommit = "4d1bda0016c42bbe9c0470ca976f10cf6774fd8a";' in text
    assert 'amgxGitUrl = "https://github.com/NVIDIA/AMGX.git";' in text
    assert (
        'amgxSourceHash = "sha256-XKyGG1wsG37qlSTukZMl8BKyi248SCQKHdlgVYfnR6A=";'
        in text
    )
    assert "fetchSubmodules = true;" in text
    assert "amgxSourceArchive = if linuxCudaSupported then pkgsCuda.runCommand" in text
    assert "chmod -R u+rwX,go+rX" in text
    assert "cuda_nvtx.include}/include/nvToolsExt.h" in text
    assert "libcurand.include}/include/curand*.h" in text
    assert text.count('"${pkgsCuda.openmpi}/lib/libmpi.so"') >= 6
    assert "target_link_libraries(amgxsh CUDA::cublas" in text
    assert "target_link_libraries(amgx_tests_launcher amgxsh" in text
    assert (
        "lib.optionals withAmgx [\n                  pkgsCuda.cudaPackages.cuda_nvtx"
        in text
    )
    assert "pkgsCuda.cudaPackages.libcurand" in text
    assert "cudaPetscAmgx = mkCudaPetsc { withAmgx = true; };" in text
    assert 'profile = "cuda-amgx";' in text
    assert "pyeidors-cuda-amgx = pyeidorsCudaAmgx;" in text
    assert 'eit-app-cuda-amgx = mkApp "pyeidors-cuda-amgx"' in text
    assert '"cuda-amgx" = pkgsCuda.mkShell {' in text
    assert "--download-amgx=${amgxSourceArchive}" in text
    assert '"--with-64-bit-indices=0"' in text
    assert '"--with-cxx-dialect=17"' in text
    assert '"--with-cuda-dialect=17"' in text
    assert (
        'assert !withAmgx || allowComplexAmgx || scalarType == null || scalarType == "real";'
        in text
    )
    assert (
        'assert !withAmgx || allowComplexAmgx || precision == null || precision == "double";'
        in text
    )
    assert "Verify PCAMGX with a setup/solve smoke" in text
    assert "opts['pc_amgx_smoother'] = 'JACOBI_L1'" in text
    assert "opts['pc_amgx_exact_coarse_solve'] = '0'" in text
    assert "pc.setType('amgx'); print(pc.getType())" not in text


def test_native_complex_amgx_profile_is_complex128_only_experiment():
    text = _flake_text()

    assert "allowComplexAmgx ? false" in text
    assert "AMGX_mode_dZZI" in text
    assert "AMGX_mode_dCCI" in text
    assert "PetscOptionsReal" in text
    assert "cudaPetscComplexAmgx = mkCudaPetsc {" in text
    assert "allowComplexAmgx = true;" in text
    assert 'profile = "complex-cuda-amgx";' in text
    assert "pyeidors-complex-cuda-amgx = pyeidorsComplexCudaAmgx;" in text
    assert 'eit-app-complex-cuda-amgx = mkApp "pyeidors-complex-cuda-amgx"' in text
    assert '"complex-cuda-amgx" = pkgsCuda.mkShell {' in text
    assert 'export PYEIDORS_PETSC_SCALAR_TYPE="complex"' in text
    assert 'export PYEIDORS_PETSC_AMGX_ENABLED="1"' in text
    assert "complex64-cuda-amgx" not in text
    assert "cudaPetscComplexSingleAmgx" not in text

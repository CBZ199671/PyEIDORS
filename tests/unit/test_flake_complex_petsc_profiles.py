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
    assert 'envProfile = "complex-cuda";' in text
    assert 'export PYEIDORS_PETSC_SCALAR_TYPE="complex"' in text
    assert 'export PYEIDORS_PETSC_DEVICE_DEFAULT="cuda"' in text
    assert "cudaPetscComplex" in text
    assert "cudaFenicsDolfinxComplex" in text

    assert '"complex64-cuda" = pkgsCuda.mkShell {' in text
    assert 'envProfile = "complex64-cuda";' in text
    assert 'export PYEIDORS_PETSC_SCALAR_TYPE="complex64"' in text
    assert "cudaPetscComplexSingle" in text
    assert "cudaFenicsDolfinxComplexSingle" in text

# Phase-2 Hard Cutover Migration Notes

This document lists the intentional breaking changes introduced by the Phase-2 migration to the FEniCSx-only runtime.

## Breaking API Changes

| Area | Old (removed) | New (supported) |
|---|---|---|
| Environment info | `check_environment()["fenics_available"]` | `check_environment()["dolfinx_available"]` |
| Gauss-Newton class | `StandardGaussNewtonReconstructor` / `ModularGaussNewtonReconstructor` | `GaussNewtonReconstructor` |
| Mesh loader API | `load_fenics_mesh()` | `MeshLoader.load_mesh()` / `MeshLoader.get_default_mesh()` |
| Mesh object behavior | dynamic monkey-patched mesh attributes | explicit `EITMesh` dataclass container |
| Function array access | `Function.vector()` compatibility layer | `Function.x.array` and `femx.function_get_array()` helpers |
| Demo script names | `demo_fenics_*` | `demo_dolfinx_*` |

## Runtime Behavior Changes

1. PyEIDORS now hard-targets DOLFINx APIs. Legacy `fenics/dolfin` imports are blocked by CI guardrails.
2. Mesh cache loading is `.msh` + DOLFINx-native pipeline first.
3. CI enforces:
   - legacy token guard (`scripts/ci/legacy_guard.py`)
   - mandatory gmsh test path
   - coverage gate `--cov-fail-under=80`

## Upgrade Checklist

1. Update your imports to FEniCSx/DOLFINx pathways.
2. Replace removed symbol names listed above.
3. Regenerate/verify mesh caches as `.msh`.
4. Run full validation locally:

```bash
nix develop -c bash -lc 'python -m pytest -q --cov=src/pyeidors --cov-fail-under=80'
```

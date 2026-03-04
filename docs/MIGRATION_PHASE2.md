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
| System setup | implicit mesh fallback inside `EITSystem.setup()` | explicit `setup(mesh=...)` / `setup(mesh_source='cache'|'generated', ...)` |
| Solver return contract | ad-hoc `dict`/`Function` mixed outputs | typed `SolverOutput` for GN/Sparse solvers |
| Forward backend | SciPy-only practical path | PETSc default (`linear_backend='petsc'`) + SciPy fallback |
| Cache architecture | ad-hoc script-level reuse | two-layer cache (`process` + `disk`) via `CacheManager` |

## Runtime Behavior Changes

1. PyEIDORS now hard-targets DOLFINx APIs. Legacy `fenics/dolfin` imports are blocked by CI guardrails.
2. Mesh cache loading is `.msh` + DOLFINx-native pipeline first.
3. CI enforces:
   - legacy token guard (`scripts/ci/legacy_guard.py`)
   - mandatory gmsh test path
   - coverage gate `--cov-fail-under=85`
   - perf guard (`scripts/ci/perf_snapshot.py` + `scripts/ci/perf_guard.py`)
4. New runtime knobs:
   - `EITSystem(..., cache_scope='both', cache_dir='.pyeidors_cache/v2')`
   - `EITSystem(..., performance_mode='aggressive'|'safe')`
   - `EITSystem(..., linear_backend='petsc'|'scipy')`
5. Cache runtime API:
   - `system.get_cache_stats()`
   - `system.clear_cache(scope='process'|'disk'|'both')`

## Upgrade Checklist

1. Update your imports to FEniCSx/DOLFINx pathways.
2. Replace removed symbol names listed above.
3. Regenerate/verify mesh caches as `.msh`.
4. Run full validation locally:

```bash
nix develop -c bash -lc 'python -m pytest -q --cov=src/pyeidors --cov-fail-under=85'
```

### Performance Guard Locally

```bash
nix develop -c bash -lc '
  python scripts/ci/perf_snapshot.py --mode baseline --repeat 5 --output test_results/perf/baseline.json
  python scripts/ci/perf_snapshot.py --mode optimized --repeat 5 --output test_results/perf/optimized.json
  python scripts/ci/perf_guard.py \
    --baseline test_results/perf/baseline.json \
    --optimized test_results/perf/optimized.json \
    --report test_results/perf/comparison.md \
    --min-improvement 0.50 --max-regression 0.05
'
```

See also:
- `docs/CACHE_ARCHITECTURE.md`

# Cache Architecture (EIDORS-Style Semantic Cache)

PyEIDORS uses a two-layer cache inspired by EIDORS `eidors_cache`, with semantic
dependency signatures, deterministic invalidation, and rank-aware eviction.

## Layers

1. **L1 Process cache**
   - In-memory rank-aware store.
   - `score_eff = round(10*log10(effort * use_count)) + priority`
   - `score_size = round(10*log10(size_bytes / 1024))`
   - Eviction removes low-priority entries first via key:
     - retention rank: `(-score_eff, score_size, -last_access)`
     - eviction rank: `(score_eff, -score_size, last_access)`
   - Optimized for repeated solves in one Python process.
   - Default size budget: `3 GB`.

2. **L2 Disk cache**
   - Persistent object store under `.pyeidors_cache/v2`.
   - sqlite index (`index.sqlite`) + object payload files.
   - Maintains `name/namespace/effort/use_count/priority/score_eff/score_size/score` metadata.
   - Default size budget: `20 GB`.

## Artifact kinds

- `mesh_bundle`
- `pattern_bundle`
- `forward_factor`
- `jacobian`
- `single_step_operator`
- `sparse_basis`
- `measurement_projection`

## Key design

Keys are SHA-256 hashes generated from:

- `cache_schema_version` (current: `2`)
- artifact kind
- semantic payload (mesh, pattern, drive mode/value, backend config, etc.)
- code fingerprint

Any relevant model/backend/physics change produces a new key, preventing stale reuse.

For EIDORS-style shorthand usage, PyEIDORS also supports semantic cache objects via
`CacheManager.get_or_compute_semantic(...)`, where keys are derived from normalized
dependency signatures (`cache_obj_signature`) rather than runtime object identity.

## Invalidation rules

Invalidate automatically by key mismatch when any of the following changes:

- mesh geometry / tags / association
- pattern config
- drive configuration and geometry scale
- contact impedance
- backend solver config
- code fingerprint / cache schema
- background conductivity (`sigma_hash`)
- Jacobian payload hash
- linear backend config changes (PETSc/SciPy solver options)

Invalidate manually by management API:

- `clear_name(name, namespace=None)`
- `clear_max(max_bytes)`
- `clear_old(timestamp)`
- `clear_new(timestamp)`

Manual invalidation:

```python
system.clear_cache(scope="both")
```

Additional EIDORS-like operations:

- `cache_manager.clear_name(name, namespace=None)`
- `cache_manager.clear_max(max_bytes)`
- `cache_manager.clear_old(timestamp)`
- `cache_manager.clear_new(timestamp)`
- `cache_manager.collect_recent(names=[...], limit_per_name=1, include_value=False)`
- `cache_manager.install_to_cache(snapshot, target_layers="both")`
- `cache_manager.status(name=None)` / `cache_manager.set_enabled(on, name=None)`
- `cache_manager.debug_status(name=None)` / `cache_manager.set_debug(on, name=None)`
- `cache_manager.boost_priority(delta)`

## Runtime API

`EITSystem` exposes:

- `get_cache_stats() -> dict`
- `clear_cache(scope="process"|"disk"|"both")`

Stats include hit/miss counters and process/disk footprint.
Stats also include artifact and namespace breakdown for each layer.
Stats also include global cache/debug status, disabled function names, and active priority boost.

## Corruption handling

If a disk payload is unreadable/corrupted:

1. the entry is removed automatically
2. computation falls back to recompute
3. workflow continues without hard failure

## Performance notes

- Forward solve caches matrix factors (`forward_factor`) for repeated same-sigma solves.
- Jacobian and sparse basis reuse are enabled through the same manager interface.
- Single-step difference reconstruction caches `J/Jᵀ/NOSER/A(LU)` via
  `single_step_operator` and reuses them across runs when background conductivity and
  model signatures are unchanged.

## EIDORS Mapping

| EIDORS command | PyEIDORS equivalent |
|---|---|
| `eidors_cache(@func, {args}, opt.cache_obj, opt.fstr)` | `get_or_compute_semantic(..., name=fstr, cache_obj=...)` |
| `clear_name` | `cache_manager.clear_name(name, namespace)` |
| `clear_max` | `cache_manager.clear_max(max_bytes)` |
| `clear_old` / `clear_new` | `cache_manager.clear_old(ts)` / `cache_manager.clear_new(ts)` |
| `collect_recent` | `cache_manager.collect_recent(names, ..., include_value=True)` |
| `install_to_cache` | `cache_manager.install_to_cache(snapshot, target_layers)` |
| `on/off` (global or per function) | `cache_manager.set_enabled(on, name=None|func_name)` |
| `debug_on/debug_off` | `cache_manager.set_debug(on, name=None|func_name)` |
| `boost_priority` | `cache_manager.boost_priority(delta)` |

For command-line operations, use:

- `python scripts/cache/cache_ctl.py status`
- `python scripts/cache/cache_ctl.py off --name calc_jacobian`
- `python scripts/cache/cache_ctl.py on --name calc_jacobian`
- `python scripts/cache/cache_ctl.py clear-old --timestamp <epoch-seconds>`
- `python scripts/cache/cache_ctl.py collect-recent --name inv_solve_diff_GN_one_step --with-values --output snapshot.json`
- `python scripts/cache/cache_ctl.py install-to-cache --input snapshot.json --target-layers both`

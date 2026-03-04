# Cache Architecture (Phase-2 Hard Refactor)

PyEIDORS now uses a two-layer cache inspired by EIDORS `eidors_cache`, with semantic
dependency signatures, deterministic invalidation, and score-aware eviction.

## Layers

1. **L1 Process cache**
   - In-memory score-aware store (`score = log10(effort * use_count) + priority`).
   - Optimized for repeated solves in one Python process.
   - Default size budget: `3 GB`.

2. **L2 Disk cache**
   - Persistent object store under `.pyeidors_cache/v2`.
   - sqlite index (`index.sqlite`) + object payload files.
   - Maintains `name/namespace/effort/use_count/priority/score` metadata.
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

Manual invalidation:

```python
system.clear_cache(scope="both")
```

Additional EIDORS-like operations:

- `cache_manager.clear_name(name, namespace=None)`
- `cache_manager.clear_max(max_bytes)`
- `cache_manager.collect_recent(names=[...], limit_per_name=1)`

## Runtime API

`EITSystem` exposes:

- `get_cache_stats() -> dict`
- `clear_cache(scope="process"|"disk"|"both")`

Stats include hit/miss counters and process/disk footprint.
Stats also include artifact and namespace breakdown for each layer.

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

- `eidors_cache(@func, cache_obj, fstr)`:
  - PyEIDORS equivalent: `get_or_compute_semantic(..., name=fstr, cache_obj=...)`.
- `clear_name`:
  - PyEIDORS equivalent: `cache_manager.clear_name(...)`.
- `clear_max`:
  - PyEIDORS equivalent: `cache_manager.clear_max(...)`.
- `effort/count/priority`:
  - Stored on each entry and used by score-aware eviction in both process and disk layers.

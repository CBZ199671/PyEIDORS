# Cache Architecture (Phase-2 Hard Refactor)

PyEIDORS now uses a two-layer cache inspired by EIDORS `eidors_cache`, with explicit
keys and deterministic invalidation.

## Layers

1. **L1 Process cache**
   - In-memory LRU store.
   - Optimized for repeated solves in one Python process.
   - Default size budget: `3 GB`.

2. **L2 Disk cache**
   - Persistent object store under `.pyeidors_cache/v2`.
   - sqlite index (`index.sqlite`) + object payload files.
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

## Runtime API

`EITSystem` exposes:

- `get_cache_stats() -> dict`
- `clear_cache(scope="process"|"disk"|"both")`

Stats include hit/miss counters and process/disk footprint.

## Corruption handling

If a disk payload is unreadable/corrupted:

1. the entry is removed automatically
2. computation falls back to recompute
3. workflow continues without hard failure

## Performance notes

- Forward solve caches matrix factors (`forward_factor`) for repeated same-sigma solves.
- Jacobian and sparse basis reuse are enabled through the same manager interface.
- Single-step difference scripts cache linear operators/factors to accelerate batch runs.


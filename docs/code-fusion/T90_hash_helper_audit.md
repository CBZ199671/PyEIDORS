# T90 — Hash helper audit

§V cites: V36, V62, V65, V67, V76, T79, T82.

## §1 Scope

Inventory `hashlib.sha256` call sites in `src/pyeidors/` + `scripts/`.
Classify into kinds. Decide which (if any) can swap to
`pyeidors.cache.keys.hash_array` without breaking persisted-artifact
contracts. Out of scope: actual replacement (no schema bump in T90).

## §2 Inventory baseline

`grep -rn 'hashlib\.sha256(' src/pyeidors/ scripts/` → 35 sites total
(29 in `src/pyeidors/`, 6 in `scripts/`).

`grep -rn 'hashlib\.sha256(.*\.tobytes())' src/pyeidors/` → 0 direct
raw-`arr.tobytes()` digests. Current multi-line raw-tobytes digests in
`src/pyeidors/` are none. Per-file totals are:

| file | total `sha256(` | raw `arr.tobytes()` (single+multi) | role |
|---|---|---|---|
| `cache/keys.py` | 6 | 1 (canonical `hash_array`) | A canonical |
| `cache/object_signature.py` | 2 | 0 | A canonical |
| `cache/disk_artifacts.py` | 2 | 0 | C schema-locked + D file |
| `cache/process_lru.py` | 1 | 0 | C schema-locked |
| `forward/eit_forward_model.py` | 3 | 0 | C + D |
| `forward/cuda_structured_backend.py` | 0 | 0 | migrated to `hash_array_payload` |
| `inverse/jacobian/linearized.py` | 0 | 0 | migrated to `hash_array_payload` |
| `inverse/jacobian/direct_jacobian.py` | 0 | 0 | migrated to `hash_array_payload` |
| `inverse/solvers/sparse_bayesian_engine.py` | 0 | 0 | migrated to `hash_array_payload` |
| `inverse/solvers/gauss_newton_linear_system.py` | 2 | 0 | migrated payload hashes to `hash_array_payload`; remaining C JSON digests |
| `inverse/solvers/gauss_newton_startup_cache.py` | 0 | 0 | migrated to `hash_array_payload` |
| `inverse/reduced/snapshot_bank.py` | 0 | 0 | migrated to `hash_array_payload` |
| `inverse/prior/rtr.py` | 3 | 0 | C schema-locked |
| `inverse/prior/tv_irls.py` | 1 | 0 | C schema-locked |
| `inverse/reconstruction_matrix.py` | 2 | 0 | C V36 RM signature |
| `inverse/greit.py` | 2 | 0 | C V62 GREIT signature |
| `inverse/greit_registry.py` | 1 | 0 | B migrated; remaining C GREIT registry |
| `interop/protocol_mapping.py` | 1 | 0 | C Bridge v3 runtime proof |
| `io/hdf5_artifacts.py` | 2 | 0 | C HDF5 metadata + streaming dataset payload digest |
| `perf/capabilities.py` | 1 | 0 | C PETSc CUDA probe disk-cache key |

Scripts/benchmarks/env (6 sites): `benchmark_difference_runtime.py:339`,
`benchmark_mesh_io_formats.py:471`, `benchmark_greit_eidors_parity_48e.py:841`,
`run_synthetic_parity.py:616`, `common/gn_difference_runner.py:1559`,
`env/export_env_manifest.py:66`.
All F (report-only) or D (file).

## §3 Classification

- **A canonical** — already implements / is canonical helper. ⊥ migration.
  - `cache/keys.py:24,73,82,88,96` (`hash_array`, `build_cache_key`, `hash_path`)
  - `cache/keys.py:18`, `cache/object_signature.py:45,77`
- **B migration candidate (raw `arr.tobytes()` cache key)** — none remain in
  `src/pyeidors/`; migrated candidates keep byte-stable legacy payload digests.
- **B migrated with byte-stable streaming** — still hash the exact legacy
  payload bytes, but feed them through `cache.keys.hash_array_payload` to avoid
  a full `.tobytes()` copy.
  - `inverse/jacobian/linearized.py:49-64` `compute_sigma_fingerprint`
  - `inverse/jacobian/direct_jacobian.py:334-342` direct-Jacobian `sigma_hash`
  - `inverse/solvers/gauss_newton_startup_cache.py:35-49` startup `sigma_hash`
  - `inverse/solvers/gauss_newton_linear_system.py` sparse-csr
    regularization fingerprint, dense regularization fallback, ROM
    `snapshot_hash` / `jacobian_hash` / `basis_hash`
  - `forward/cuda_structured_backend.py` `_stable_hash`
  - `inverse/solvers/sparse_bayesian_engine.py` `baseline_hash`
  - `inverse/reduced/snapshot_bank.py` snapshot matrix/column dedupe hashes
  - `inverse/greit_registry.py` ndarray signature payload hash
- **C schema-locked encoded-payload digests** — input is
  `json.dumps(payload, sort_keys=True).encode()` not ndarray.
  `hash_array` ! applicable; these stay on `hashlib.sha256(encoded)`.
  Internal payload normalization already routes ndarray fields through
  `cache.keys._normalize` / `cache.object_signature._normalize_for_signature`
  which uses `hash_array` for ndarray contents. No migration needed.
  Bridge v3 protocol/current mapping proofs in
  `interop/protocol_mapping.py` likewise hash a canonical JSON payload so the
  fingerprint proves the exact channel mapping and runtime current scaling.
- **D file content / streaming** — `read_bytes()` / chunked
  `digest.update(chunk)`. Out of scope for `hash_array`.
  - `cache/keys.py:82` `hash_path`
  - `cache/disk_artifacts.py:138` artifact file digest
  - `forward/eit_forward_model.py:193` mesh streaming hasher
    (mixes coordinates + connectivity + association + electrode payload
    via `hasher.update` chain — not a single-array digest; ! file-style)
  - `io/hdf5_artifacts.py` dataset digest helper streams HDF5 numeric payloads
    with dtype/shape framing to preserve artifact checksum semantics without
    full dataset materialization.
  - `scripts/env/export_env_manifest.py:66` env file digest
- **E bytes wrappers** — `__bytes__` payload normalization within
  `_normalize`. Already canonical.
  - `cache/keys.py:18`, `cache/object_signature.py:45`
- **F report-only / scripts** — benchmark run-id / parity report
  digests. Migration cosmetic, no benefit.
  - `scripts/benchmarks/...` 3 sites + `run_synthetic_parity.py:616`
    + `common/gn_difference_runner.py:1460`

## §4 V76 semantic-cache check

V76 forbids `id(obj)` memoization over mutable signature inputs. Audit
of every B-class site:

- All B-class digests derive from array payload bytes (content) — not `id`.
  Migrated sites use byte-stable `hash_array_payload`.
- All cache-key payloads embed those digests in JSON before final
  `stable_signature_hash`/`build_cache_key` — content-safe.
- Cholesky cache (`gauss_newton_linear_system.py:1385`) keys on JSON
  payload containing streamed sparse-csr fingerprints → content-safe.
- ROM caches embed streamed `snapshot_hash` / `jacobian_hash` /
  `basis_hash` payload digests → content-safe.
- T88 already added gate `tests/unit/test_cache_semantic_signature.py`
  + invariant V76 against id-only memoization regressions in
  `pyeidors.cache.object_signature`.

⇒ V76 holds across all current B-class cache layers. No remediation
required from T90.

## §5 Recommendations

### Safe-now (no schema bump)

1. Done under V248: sigma-hash sites in `linearized.py`,
   `direct_jacobian.py`, and `gauss_newton_startup_cache.py` now use
   `hash_array_payload`, preserving legacy real/complex digest bytes without
   schema bump.
2. Done under V250: `gauss_newton_linear_system.py` cache payload hashes now
   use `hash_array_payload`, preserving legacy sparse/dense/ROM digest bytes
   without schema bump.
3. Done under V251: remaining raw array digest sites in
   `cuda_structured_backend.py`, `snapshot_bank.py`,
   `sparse_bayesian_engine.py`, and `greit_registry.py` now use
   `hash_array_payload`, preserving legacy payload bytes without schema bump.

### Defer (require coordinated schema bump + golden refresh)

4. Future replacement with `hash_array` would add dtype/shape prefixes to
   these migrated legacy payload digests and therefore changes the cache key
   contract for relevant `pyeidors.cache` artifacts — every saved
   `*-cache.h5` / `cache_manager` blob would invalidate on first read.
   ! coordinate with `CacheKeyParts.schema_version` bump (`keys.py:80`,
   currently `= 2`) + golden fixture refresh under
   `tests/fixtures/sweep_hdf5_tables/` + V36/V62 RM/GREIT signature
   re-baseline. Out of T90 scope.

### Leave (canonical)

5. A/C/D/E/F — already canonical, schema-locked encoded payloads, file
   digests, byte wrappers, or report-only. No migration applies.

## §6 Gate

- `tests/unit/test_t90_hash_audit_gate.py` locks per-file
  `hashlib.sha256(` count for `src/pyeidors/`. Any new addition
  without audit-doc update trips the gate.
- Existing `tests/unit/test_cache_semantic_signature.py` (V76) keeps
  guarding semantic cache invariants.

## §7 Closure

T90 = audit-only. No code change in `src/`. Future migrations land via
new tasks (`T93+` tentative) once a cache-schema-bump window opens.

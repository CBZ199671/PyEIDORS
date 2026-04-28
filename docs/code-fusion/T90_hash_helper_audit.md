# T90 — Hash helper audit

§V cites: V36, V62, V65, V67, V76, T79, T82.

## §1 Scope

Inventory `hashlib.sha256` call sites in `src/pyeidors/` + `scripts/`.
Classify into kinds. Decide which (if any) can swap to
`pyeidors.cache.keys.hash_array` without breaking persisted-artifact
contracts. Out of scope: actual replacement (no schema bump in T90).

## §2 Inventory baseline

`grep -rn 'hashlib\.sha256(' src/pyeidors/ scripts/` → 49 sites total
(43 in `src/pyeidors/`, 6 in `scripts/`).

`grep -rn 'hashlib\.sha256(.*\.tobytes())' src/pyeidors/` → 10 single-line
raw-`arr.tobytes()` digests. Multi-line raw-tobytes digests live at
`forward/eit_forward_model.py:1475,1478`,
`inverse/solvers/gauss_newton_linear_system.py:967,970,973,1495,1562,1677,1680`,
`inverse/solvers/gauss_newton_startup_cache.py:38`. Adding multi-line
hits the per-file totals are:

| file | total `sha256(` | raw `arr.tobytes()` (single+multi) | role |
|---|---|---|---|
| `cache/keys.py` | 6 | 1 (canonical `hash_array`) | A canonical |
| `cache/object_signature.py` | 2 | 0 | A canonical |
| `cache/disk_artifacts.py` | 2 | 0 | C schema-locked + D file |
| `cache/process_lru.py` | 1 | 0 | C schema-locked |
| `forward/eit_forward_model.py` | 5 | 3 (1391, 1475, 1478) | B + C + D |
| `forward/cuda_structured_backend.py` | 1 | 1 | B (in-memory) |
| `inverse/jacobian/linearized.py` | 1 | 1 | B (V9 guard) |
| `inverse/jacobian/direct_jacobian.py` | 1 | 1 | B (jacobian cache key) |
| `inverse/solvers/sparse_bayesian_engine.py` | 1 | 1 | B (SVD reuse cache) |
| `inverse/solvers/gauss_newton_linear_system.py` | 10 | 8 (967,970,973,987,1495,1562,1677,1680) | B + C |
| `inverse/solvers/gauss_newton_startup_cache.py` | 1 | 1 | B (startup cache) |
| `inverse/reduced/snapshot_bank.py` | 3 | 3 (42, 105, 108) | B (in-memory dedup) |
| `inverse/prior/rtr.py` | 3 | 0 | C schema-locked |
| `inverse/prior/tv_irls.py` | 1 | 0 | C schema-locked |
| `inverse/reconstruction_matrix.py` | 2 | 0 | C V36 RM signature |
| `inverse/greit.py` | 2 | 0 | C V62 GREIT signature |
| `io/hdf5_artifacts.py` | 1 | 0 | C HDF5 metadata |

Scripts/benchmarks (6 sites): `benchmark_difference_runtime.py:339`,
`benchmark_mesh_io_formats.py:471`, `benchmark_greit_eidors_parity_48e.py:841`,
`run_synthetic_parity.py:616`, `common/gn_difference_runner.py:1460`,
`env/export_env_manifest.py:66`. All F (report-only) or D (file).

## §3 Classification

- **A canonical** — already implements / is canonical helper. ⊥ migration.
  - `cache/keys.py:24,73,82,88,96` (`hash_array`, `build_cache_key`, `hash_path`)
  - `cache/keys.py:18`, `cache/object_signature.py:45,77`
- **B migration candidate (raw `arr.tobytes()` cache key)** — embedding raw
  digest into `cache_key` payload. Replacing with `hash_array` adds
  dtype/shape prefix → digest changes → cache key changes → on-disk
  artifacts invalidate. ! schema/version bump + golden refresh before
  swap.
  - `forward/eit_forward_model.py:1391` `_sigma_fingerprint` (transient)
  - `forward/eit_forward_model.py:1475,1478` `z_hash`, `pattern_hash`
    (embedded in model_signature payload, persisted via
    `stable_signature_hash`)
  - `forward/cuda_structured_backend.py:181` `_stable_hash` (process-local)
  - `inverse/jacobian/linearized.py:39` `compute_sigma_fingerprint`
    (V9 permissive guard, in-memory)
  - `inverse/jacobian/direct_jacobian.py:315` `sigma_hash` →
    `cache_manager.get_or_compute_semantic` artifact key
  - `inverse/solvers/sparse_bayesian_engine.py:178` `baseline_hash` →
    SVD reuse cache
  - `inverse/solvers/gauss_newton_linear_system.py:967-987` sparse-csr
    fingerprint (indptr/indices/data) + dense fallback
  - `inverse/solvers/gauss_newton_linear_system.py:1495,1562,1677,1680`
    ROM cache keys (snapshot_hash, jacobian_hash, basis_hash)
  - `inverse/solvers/gauss_newton_startup_cache.py:38` startup `sigma_hash`
  - `inverse/reduced/snapshot_bank.py:42,105,108` snapshot dedupe
    (in-memory only — see §5 safe-now)
- **C schema-locked encoded-payload digests** — input is
  `json.dumps(payload, sort_keys=True).encode()` not ndarray.
  `hash_array` ! applicable; these stay on `hashlib.sha256(encoded)`.
  Internal payload normalization already routes ndarray fields through
  `cache.keys._normalize` / `cache.object_signature._normalize_for_signature`
  which uses `hash_array` for ndarray contents. No migration needed.
- **D file content / streaming** — `read_bytes()` / chunked
  `digest.update(chunk)`. Out of scope for `hash_array`.
  - `cache/keys.py:82` `hash_path`
  - `cache/disk_artifacts.py:138` artifact file digest
  - `forward/eit_forward_model.py:193` mesh streaming hasher
    (mixes coordinates + connectivity + association + electrode payload
    via `hasher.update` chain — not a single-array digest; ! file-style)
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

- All B-class digests derive from `arr.tobytes()` (content) — not `id`.
- All cache-key payloads embed those digests in JSON before final
  `stable_signature_hash`/`build_cache_key` — content-safe.
- Cholesky cache (`gauss_newton_linear_system.py:1147`) keys on JSON
  payload containing the sparse-csr fingerprint at 967/970/973 →
  content-safe.
- ROM caches (1495/1562/1677/1680) embed `snapshot_hash` /
  `jacobian_hash` / `basis_hash` from raw tobytes → content-safe.
- T88 already added gate `tests/unit/test_cache_semantic_signature.py`
  + invariant V76 against id-only memoization regressions in
  `pyeidors.cache.object_signature`.

⇒ V76 holds across all current B-class cache layers. No remediation
required from T90.

## §5 Recommendations

### Safe-now (no schema bump)

1. `inverse/reduced/snapshot_bank.py:42,105,108` — in-memory dedup,
   never persisted. Migrating to `hash_array` only changes per-call
   set membership, not artifacts. Optional cleanup; can land in any
   later code-fusion task without §V impact.

### Defer (require coordinated schema bump + golden refresh)

2. All other B-class sites (forward `_sigma_fingerprint`/`z_hash`/
   `pattern_hash`, jacobian/sparse/SVD/ROM/startup caches, sparse-csr
   fingerprint trio at 967-973). Replacement changes the cache key
   contract for `pyeidors.cache` artifacts — every saved
   `*-cache.h5` / `cache_manager` blob would invalidate on first read.
   ! coordinate with `CacheKeyParts.schema_version` bump (`keys.py:50`,
   currently `= 2`) + golden fixture refresh under
   `tests/fixtures/sweep_hdf5_tables/` + V36/V62 RM/GREIT signature
   re-baseline. Out of T90 scope.

### Leave (canonical)

3. A/C/D/E/F — already canonical, schema-locked encoded payloads, file
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

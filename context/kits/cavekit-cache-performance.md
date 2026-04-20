---
status: draft
source: from-code
domain: cache-performance
---

# Cavekit: Cache and Performance

## Scope

This kit covers process and disk caches, semantic cache keys, invalidation,
session lifecycle, performance policy, and benchmark gates.

## Requirements

### R1: Semantic cache keys prevent stale reuse

**Description:** Cached artifacts are keyed by semantic inputs, code schema, and
backend configuration so incompatible artifacts are not reused.

**Acceptance Criteria:**
- [ ] Mesh, pattern, drive, contact impedance, backend config, conductivity, and
  code schema changes produce distinct keys.
- [ ] Object-signature tests remain stable across equivalent semantic inputs.
- [ ] Cache corruption removes the bad entry and recomputes.

**Dependencies:** `cavekit-core-system.md`

### R2: Process and disk cache layers have observable lifecycle

**Description:** Users can inspect, clear, collect, install, enable, disable, and
debug cache entries by name and scope.

**Acceptance Criteria:**
- [ ] Process cache respects size and score-aware eviction policy.
- [ ] Disk cache uses indexed payload storage and reports artifact/namespace
  breakdown.
- [ ] Session-scoped disk caches clean only the owning session.

**Dependencies:** `cavekit-environment-cli.md`

### R3: Performance policy selects safe defaults

**Description:** Strict, fast, GPU, fused, and reduced paths are selected through
documented policy rather than implicit environment guessing.

**Acceptance Criteria:**
- [ ] Policy helpers normalize acceleration profile and backend options.
- [ ] Benchmark guards compare reports against expected thresholds or parity
  rules.
- [ ] Experimental paths remain opt-in unless promoted by policy.

**Dependencies:** `cavekit-forward-solver.md`, `cavekit-inverse-reconstruction.md`

### R4: Benchmark scripts emit reproducible reports

**Description:** Benchmark and diagnostics scripts produce machine-readable
reports usable for regression gates.

**Acceptance Criteria:**
- [ ] CUDA and CPU benchmark reports include effective solver/backend fields.
- [ ] Report comparison scripts detect regressions or mismatched fields.
- [ ] Performance guard tests cover helper behavior and report parsing.

**Dependencies:** `cavekit-environment-cli.md`

## Brownfield Evidence

- Source: `src/pyeidors/cache/`
- Source: `src/pyeidors/perf/`
- Source: `scripts/cache/cache_ctl.py`
- Source: `scripts/benchmarks/`
- Docs: `docs/CACHE_ARCHITECTURE.md`
- Tests: `tests/unit/test_cache_manager_extended.py`
- Tests: `tests/unit/test_cache_semantic_signature.py`
- Tests: `tests/unit/test_perf_policy.py`
- Tests: `tests/integration/test_3d_diff_perf_gate.py`

## Out of Scope

- Algorithm-specific numerical correctness; see Forward Solver and Inverse
  Reconstruction.

## Cross-References

- Depends on: `cavekit-core-system.md`
- Related: `cavekit-forward-solver.md`
- Related: `cavekit-inverse-reconstruction.md`
- Related: `cavekit-environment-cli.md`


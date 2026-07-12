# Dynamic Kalman Sequence Acceptance

Run the deterministic measurement-space acceptance suite from the repository
root:

```bash
nix develop -c uv run python -m eit_app.dynamic_acceptance \
  --output output/ecd-cwr-dynamic-acceptance/pyeidors-dynamic-acceptance.json
```

Run the report builder in-process under explicit source coverage:

```bash
nix develop -c uv run pytest -q -o addopts="" \
  tests/unit/test_dynamic_acceptance.py \
  --cov=eit_app.dynamic_acceptance --cov-report=term-missing \
  --cov-fail-under=87
```

The suite exercises the production
`PersistentMeasurementDiagonalKalmanSession` with positive and negative
isolated spikes, a sustained step, a three-frame pulse, a continuous ramp, a
biphasic response, and a missing-block gap. It also checks candidate-only NIS
gating, non-candidate step preservation, session reset isolation, and the
`lag=0` backend metadata contract. Version 2 also verifies that every accepted
measurement-space update is fused with the same-frame static NOSER image with
at least `0.75` anchor gain. Runtime spatial RMS/robust-spread divergence then
returns the static NOSER image and removes the contaminated session.

Acceptance thresholds:

- end-to-end isolated-spike suppression at least 90%;
- steady-state step bias below 5%;
- ramp/biphasic peak-time error at most two blocks;
- total latency exactly two blocks (`2` upstream centered blocks + `0` backend);
- block gaps are represented by `block_step` without unordered state reuse.
- measurement-space state remains within the NOSER-anchor acceptance bound.

The isolated-spike case models the actual two-stage runtime path. The static
inverse first consumes the temporal measurement weight, then the measurement
Kalman update uses the same weight to inflate `R_eff`. Testing the Kalman layer
with an unweighted static image would not represent the EitHost pipeline.

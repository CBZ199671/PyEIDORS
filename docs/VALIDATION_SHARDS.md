# Validation Shards

This project uses sharded pytest commands for recoverable validation. The full
`tests/unit` suite has timed out before without returning enough failure detail,
so broad validation should run named shards and persist one log per shard.

Always invoke the runner through the FEniCSx/Nix environment:

```bash
nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --list
```

Print the exact default broad commands without running them. This includes GUI and
focused smoke coverage, but still skips hardware unless explicitly opted in:

```bash
nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --dry-run
```

Run the focused FEniCSx/PETSc refactor smoke shard:

```bash
nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --run --shard fp-refactor-smoke
```

Run all category shards and keep going shard by shard:

```bash
nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --run --all --timeout 300
```

By default, broad selections (`--run`, `--dry-run`, and `--all`) include the
pure-software `gui` shard and skip only the opt-in `hardware` shard. GUI
assertion failures are therefore reported as GUI software regressions, not hidden
behind missing hardware. Hardware abstraction, serial transport, relay,
protocol, simulator/factory, and device discovery tests run only when explicitly
requested:

```bash
nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --run --all --include-hardware --timeout 300
```

The two shards can also be reviewed independently:

```bash
nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --run --shard gui --timeout 300
nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --run --shard hardware --timeout 300
```

Forward extra pytest arguments with repeated `--pytest-arg`. Use the
`--pytest-arg=<value>` form for dash-prefixed pytest options:

```bash
nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --dry-run --shard fp-refactor-smoke --pytest-arg=-k --pytest-arg "solver and not slow"
```

The runner writes local logs and `summary.json` under
`test_results/sharded_unit/<timestamp>/`. That directory is ignored by Git; copy
specific snippets into `context/impl/` when evidence needs to persist. Every
generated pytest command includes `--no-cov`; coverage is a separate gate.

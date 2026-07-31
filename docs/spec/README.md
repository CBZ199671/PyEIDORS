# SPEC registry

`SPEC.md` remains the required entrypoint and active queue. `registry.json` lists every authoritative document; `id-map.tsv` gives exact ID routing.

- Add active tasks and fresh bugs to the root inbox tables.
- Add invariants to the matching domain file, preserving concrete function-specific constraints.
- Never reuse or renumber V/T/B IDs.
- Move completed tasks and resolved bugs to `history/` without changing IDs.
- Treat every registered split/archive file with the same governance risk and validation standard.
- Run `python scripts/ci/spec_integrity_guard.py` after every registry edit.

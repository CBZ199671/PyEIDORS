# 001: Generate Plans From Kits

## Runtime Inputs

- Kits: `context/kits/`
- Implementation tracking: `context/impl/`
- Source directories: `src/pyeidors`, `src/eit_app`, `scripts`
- Test directories: `tests/unit`, `tests/integration`

## Context

This is the Map phase for brownfield Cavekit adoption. The code already exists.
Plans should sequence validation, documentation mapping, and gap closure before
any behavior changes.

## Task

1. Read `context/kits/cavekit-overview.md`.
2. Read `context/kits/validation-report.md`.
3. Create or update `context/plans/plan-overview.md`.
4. Create or update build-site files under `context/plans/`.
5. Assign task IDs, dependencies, file ownership, validation gates, and depth.
6. Prefer validation and source-to-kit traceability over speculative rewrites.

## Depth Rules

Use the complexity rubric from `.cavekit/config.json`.

- `quick`: documentation-only or one smoke command.
- `standard`: one domain with unit tests and small mapping changes.
- `thorough`: cross-domain validation, GUI/runtime work, CUDA/performance work,
  or tasks touching many modules.

## Exit Criteria

- [ ] `plan-overview.md` lists all build sites and task tiers.
- [ ] Each build-site task references kit requirements.
- [ ] Each task has validation gates and dependencies.
- [ ] No task asks agents to rewrite working code without a failing criterion.
- [ ] `context/impl/impl-overview.md` exists or is updated with plan status.

## Completion Signal

`<all-tasks-complete>`


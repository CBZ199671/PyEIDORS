# 002: Implement Brownfield Cavekit Plan

## Runtime Inputs

- Plan overview: `context/plans/plan-overview.md`
- Build site: `context/plans/build-site-brownfield-cavekit.md`
- Kits: `context/kits/`
- Implementation tracking: `context/impl/`

## Context

This prompt executes the brownfield Cavekit stabilization plan. The codebase
already works; prioritize validation, traceability, and accurate gap reporting.

## Task

1. Read the plan overview and build site.
2. Pick the highest-priority unblocked task.
3. Read only the kits and source files relevant to that task.
4. Execute the task's validation gates.
5. Update `context/kits/validation-report.md` and `context/impl/`.
6. If source-to-kit mappings are required, add minimal `CLAUDE.md` files.
7. Do not change production code unless a concrete failing criterion requires it
   and the user has approved that behavior change.

## Exit Criteria

- [ ] Completed task is marked in implementation tracking.
- [ ] Validation commands and outcomes are recorded.
- [ ] Any blocked task includes blocker reason and next action.
- [ ] `git diff --check` passes.

## Completion Signal

`<all-tasks-complete>`


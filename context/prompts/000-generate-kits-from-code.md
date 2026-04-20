# 000: Generate Kits From Existing Code

## Runtime Inputs

- Framework: Python package plus PySide6 desktop app.
- Source directories: `src/pyeidors`, `src/eit_app`.
- Tests: `tests/unit`, `tests/integration`.
- Docs: `README.md`, `FILE_ORGANIZATION.md`, `docs/*.md`.

## Context

This is brownfield adoption. Existing code and tests are the reference material.
The goal is not to rewrite the project; the goal is to make current behavior
explicit as agent-consumable kits.

## Task

### Phase 1: Explore

1. Read `context/refs/architecture-overview.md`.
2. Inspect package boundaries under `src/pyeidors` and `src/eit_app`.
3. Inspect tests to identify validated behavior and coverage gaps.
4. Identify domains that can be understood independently.

### Phase 2: Generate Or Update Kits

For each domain:

1. Create or update `context/kits/cavekit-{domain}.md`.
2. Keep requirements implementation-agnostic.
3. Number requirements as `R1`, `R2`, etc.
4. Give every requirement observable acceptance criteria.
5. Cite source files and tests as brownfield evidence.
6. Add out-of-scope boundaries and cross-references.

### Phase 3: Validate

1. Update `context/kits/validation-report.md`.
2. Mark criteria as covered, partially covered, or gap.
3. Do not invent passing tests. If no test exists, record a gap.

## Exit Criteria

- [ ] `context/kits/cavekit-overview.md` indexes all major domains.
- [ ] Every major source domain has at least one matching kit.
- [ ] Every requirement has testable acceptance criteria.
- [ ] Cross-domain dependencies are named.
- [ ] Validation gaps are listed.

## Completion Signal

`<all-tasks-complete>`


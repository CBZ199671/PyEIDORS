# Branching Policy

This document records the current retained branches in this repository and
defines the branch workflow to follow from now on.

## Why This Policy Exists

This repository previously accumulated many temporary branches because
AI-assisted worktree tools created one branch per task. That behavior is not
wrong by itself, but without cleanup it makes the repository noisy and makes it
hard to tell which branches are still meaningful.

The goal of this policy is simple:

- keep only a very small number of meaningful long-lived branches
- treat AI task branches as disposable
- make every retained branch explainable in one sentence
- make branch cleanup a routine part of development

## Current Retained Branches

As of 2026-04-10, the repository intentionally retains only the following local
branches.

| Branch | Role | Why it is kept now | When it can be removed |
| --- | --- | --- | --- |
| `main` | Local stable integration baseline | It is the current local pre-release baseline and already includes the environment, cache, reconstruction, documentation, and cleanup line beyond `origin/main`. | Do not remove. |
| `dev/gui-integration` | Active development mainline | This is the current working branch for the EIT workstation GUI, hardware integration, acquisition workflow, recording, reconstruction UI, and related UX polish. | Keep until this line is merged or replaced by a new primary `dev/*` branch. |
| `integration/mac-base-gpu-merge-20260309` | Specialized integration branch | This branch is a parallel integration line for the mac-freeze / GPU / WSL2 runtime convergence work. It also still has an active linked worktree, so it is still operationally meaningful. | Remove only after the cross-platform integration line is either merged, archived elsewhere, or explicitly abandoned. |
| `codex/backup-pre-unified-recon-20260304` | Historical safety snapshot | This is a pre-migration snapshot kept as a rollback anchor before the unified reconstruction hard migration. It is not an active development branch. | Remove only after the reconstruction migration is considered fully settled and no rollback anchor is needed. |

## Evidence Behind The Current Classification

The retained branches are not arbitrary. Their roles come directly from their
commit themes:

- `main`
  - recent history includes environment hardening, cache architecture changes,
    FenicsX hard cutover, documentation cleanup, and a merge from `origin/main`
  - it is ahead of `origin/main`, so it currently serves as the local stable
    baseline rather than a pure mirror of the remote default branch
- `dev/gui-integration`
  - the unique commits on top of `main` are all GUI and hardware workflow
    commits, including workstation implementation, layout changes, acquisition
    shutdown fixes, session summary relocation, and voltage fit plotting
- `integration/mac-base-gpu-merge-20260309`
  - the unique commits focus on CUDA/WSL2 hardening, mac-freeze alignment,
    exact-3D memory fallback, and related strict documentation
  - this is clearly a targeted integration campaign, not a normal day-to-day
    feature branch
- `codex/backup-pre-unified-recon-20260304`
  - it contains a single backup snapshot commit with an explicit pre-migration
    purpose in the commit message

## Branch Types Allowed Going Forward

Only the following branch families should exist long term.

### `main`

Use `main` as the local stable integration baseline.

Rules:

- do not develop experimental work directly on `main`
- only move work into `main` after it has become the accepted local baseline
- `origin/main` is allowed to lag during active development, because this
  repository can have a local pre-release phase

### `dev/*`

Use `dev/*` for the single current primary development line.

Rules:

- normally there should be only one active long-lived `dev/*` branch
- today that branch is `dev/gui-integration`
- if a new development mainline replaces it, rename or switch to a new
  `dev/<topic>` branch and retire the old one promptly

Recommended naming:

- `dev/gui-integration`
- `dev/reconstruction-runtime`
- `dev/release-prep`

### `integration/*`

Use `integration/*` only for cross-cutting merge campaigns that combine major
lines of work and genuinely need isolation.

Rules:

- every `integration/*` branch must have a clear topic and date
- every `integration/*` branch should have an explicit reason to exist outside
  `main` and `dev/*`
- if possible, it should also have a dedicated worktree
- once the integration campaign is finished, merge or archive it, then remove
  the branch

Recommended naming:

- `integration/mac-base-gpu-merge-20260309`
- `integration/release-compat-20260410`

### `backup/*`

Use `backup/*` only as a read-only rollback snapshot before risky structural
migrations.

Rules:

- never continue normal development on a `backup/*` branch
- create it immediately before a risky migration, not days later
- the branch name must say what it protects
- remove it only after the migration has proven stable for a meaningful period

Recommended naming:

- `backup/pre-unified-recon-20260304`
- `backup/pre-cache-hardcut-20260410`

## Branch Types That Should Be Disposable

Temporary AI task branches are allowed, but they must be treated as disposable.

This includes branches such as:

- `claude/*`
- `codex/*`
- any one-off task branch created only because a worktree tool required one

Rules:

- these branches are not project structure; they are execution scaffolding
- once their useful commits are merged or cherry-picked, delete them on the
  same day if possible
- do not keep them as historical decoration
- do not let them accumulate just because the tool created them automatically

If the tool forces a task branch, use it, finish the task, absorb the useful
work, then delete the branch and prune the worktree.

## Default Workflow To Follow

For this repository, use the following working pattern by default.

1. Continue normal feature development on `dev/gui-integration`.
2. If a task is small or medium, do it directly on `dev/gui-integration`.
3. If a task needs isolation, create a short-lived worktree branch.
4. Merge or cherry-pick the useful result back into `dev/gui-integration`.
5. Delete the temporary branch immediately after the result is absorbed.
6. Periodically promote stable milestones from `dev/gui-integration` into
   `main`.
7. Only keep `integration/*` or `backup/*` if they still have a real purpose.

## Cleanup Routine

Run this cleanup routinely, especially after AI-assisted sessions.

```bash
git worktree prune
git branch --merged dev/gui-integration
git branch --merged main
```

Then remove temporary branches that are already absorbed:

```bash
git branch -d <temporary-branch>
```

Good cleanup questions:

- Is this branch still actively used?
- Does it contain unique work not preserved elsewhere?
- Does its name describe a real long-lived role?
- If I delete it today, would anything operational break?

If the answer is "no" across the board, the branch should probably be removed.

## Practical Rules For This Repository

To keep this repository clean, follow these rules from now on:

- keep `main` as the local stable baseline
- keep only one long-lived active `dev/*` branch at a time
- create `integration/*` only for deliberate merge campaigns
- create `backup/*` only before risky migrations
- treat `claude/*` and `codex/*` task branches as disposable
- delete temporary branches as soon as their useful commits are absorbed
- prune stale worktrees routinely
- never keep a branch unless you can explain its purpose in one sentence

## Current Recommended Branch Roles

For day-to-day work right now, use this mental model:

- `main`: stable local baseline
- `dev/gui-integration`: current primary development line
- `integration/mac-base-gpu-merge-20260309`: specialized side integration line
- `codex/backup-pre-unified-recon-20260304`: rollback-only archive branch

If a future branch does not fit one of those roles, it probably should not be
long-lived.

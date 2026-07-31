# SPEC REGISTRY FORMAT

Root `SPEC.md` is the mandatory entrypoint. `docs/spec/registry.json` lists every
authoritative split/history document; `docs/spec/id-map.tsv` maps every ID to one
authority file. Every cavekit command reads root first, then follows the registry.

## SECTIONS

Root headers stay fixed and addressable. Registered child files hold routed rows.

```
# SPEC

## §R REGISTRY
manifest + exact ID→file lookup

## §G GOAL
one line. what code must do.

## §C CONSTRAINTS
- bullet. non-negotiable boundary.
- bullet. tech/lang/lib locked in.

## §I INTERFACES
external surface. what world sees.
- cmd: `foo bar` → stdout JSON
- api: POST /x → 200 {id}
- file: `config.yaml` schema …
- env: `FOO_KEY` required

## §V INVARIANTS
root index + inbox. numbered/testable rows live in registered domain files.
V1: ∀ req → auth check before handler
V2: token expiry ≤ ⊥ allowed
V3: DB write ! in transaction

## §T TASKS
active-only pipe table. ids monotonic (never reused). status: `x` done / `~` wip / `.` todo.
id|status|task|cites
T1|.|scaffold repo|-
T2|.|impl §I.api POST /x|V2
T3|x|add §V.1 middleware|V1,I.api

## §B BUGS
root intake table. resolved rows live in registered history. each row = bug + invariant that catches recurrence.
id|date|cause|fix
B1|2026-04-20|token `<` not `≤`|V2
B2|2026-04-21|race on write|V3
```

**Table cell rules**: literal `|` → escape as `\|`. Backticks OK. Cells trimmed. Empty = `-`.

## ADDRESSING

`§<S>.<n>` = section.item. `§V.2` = invariants section, item 2.
Commands, commits, PRs all reference by §. Zero ambiguity.

## CAVEMAN ENCODING

Default for every section. Rules:

- Drop articles (a, an, the). Drop filler.
- Drop aux verbs (is, are, was) where fragment works.
- Short synonyms (fix > implement).
- Fragments fine.

**Preserve verbatim**: code, paths, identifiers, URLs, numbers, error strings, SQL, regex.

**Symbols** (save tokens, machine-readable):

```
→   leads to / becomes / triggers
∴   therefore / fix
∀   for all / every
∃   exists / some
!   must
?   may / optional
⊥   never / impossible / forbidden
≠   not equal / differs from
∈   in / member of
∉   not in
≤   at most
≥   at least
&   and
|   or
```

**Bad** (v1 prose):

> The authentication middleware must verify the token expiry on every request before allowing the handler to execute.

**Good** (v2 caveman):

> V1: ∀ req → auth check before handler

**Bad** (prose bug note):

> Fixed a bug where token expiry comparison used strict less-than instead of less-than-or-equal, causing tokens to be rejected exactly at their expiry timestamp.

**Good** (v2 caveman):

> B1: token `<` not `<=` ∴ tokens rejected @ expiry. §V.2 now ! `≤`.

## WHY CAVEMAN FOR SPECS

Spec loaded every invocation. 75% fewer tokens = 75% fewer dollars & faster reads.
Human skims fast too. Symbols unambiguous.

## REGISTERED FILE RULE

- Root `SPEC.md` ! remain self-describing entrypoint + active queue.
- `docs/spec/registry.json` ! list every authority document.
- `docs/spec/id-map.tsv` ! exactly map every V/T/B ID to one authority file.
- §V rows → one `docs/spec/invariants/<domain>.md`; keep numeric ID + full concrete constraint.
- completed §T → `docs/spec/history/tasks-completed.md`; resolved §B → `docs/spec/history/bugs.md`.
- Split/history documents have same governance risk. Never renumber, reuse, or silently drop rows.
- After every write: `python scripts/ci/spec_integrity_guard.py` ! pass.

## WRITES

| command | writes | section |
|---|---|---|
| `/spec new` | creates | root + registry + registered files |
| `/spec amend` | edits | ID authority from `id-map.tsv` |
| `/spec bug` | appends | root §B intake + routed §V |
| `/build` | flips/moves | root §T `.` → `~`; on `x` move to completed history |
| `/check` | — | root, then all files in `registry.json` |

That is whole format.

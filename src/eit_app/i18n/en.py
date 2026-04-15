"""English translation dictionary.

Keys follow a dotted scope convention: ``<area>.<component>.<element>``.
Formatting placeholders use :py:meth:`str.format` syntax (e.g. ``{count}``).

This file is a plain Python ``dict`` (not JSON / YAML) on purpose so that:
  * IDEs can autocomplete key references,
  * ``grep`` finds usages directly,
  * static checkers can flag unused keys.

Phase 0 only seeds a tiny validation block. Real UI strings are migrated
tab-by-tab in Phase 3 (hardware / simulation / dataset / database).
"""

from __future__ import annotations

TRANSLATIONS: dict[str, str] = {
    # ------------------------------------------------------------------
    # Phase 0 validation keys — remove once Phase 3 begins populating real
    # translations.  Kept for smoke-testing the plumbing.
    # ------------------------------------------------------------------
    "_test.hello": "Hello, world!",
    "_test.greeting": "Welcome, {name}",
    "_test.plural": "You have {n} pending tasks",

    # ------------------------------------------------------------------
    # Application chrome — filled out in Phase 1 when the Language menu
    # goes in.  Included here only as a forward-looking skeleton so that
    # Phase 0's smoke test can verify end-to-end plumbing against realistic
    # keys.
    # ------------------------------------------------------------------
    "app.title": "EIT Workstation",
}

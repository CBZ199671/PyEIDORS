"""CUQI import helpers with targeted warning suppression."""

from __future__ import annotations

from contextlib import contextmanager
import warnings

_CUQI_IMPORT_WARNING_FILTERS = (
    {
        "category": UserWarning,
        "message": r"pkg_resources is deprecated as an API",
        "module": r"(pkg_resources(\..*)?|setuptools\._vendor\.pkg_resources(\..*)?|cuqi(\..*)?)",
    },
    {
        "category": PendingDeprecationWarning,
        "message": r"Importing from numpy\.matlib is deprecated",
        "module": r"(numpy\.matlib(\..*)?|cuqi(\..*)?)",
    },
)


def apply_known_cuqi_warning_filters() -> None:
    """Apply narrow warning filters for known CUQI import-time deprecations."""

    for item in _CUQI_IMPORT_WARNING_FILTERS:
        warnings.filterwarnings(
            action="ignore",
            category=item["category"],
            message=item["message"],
            module=item["module"],
        )


@contextmanager
def suppress_known_cuqi_import_warnings():
    """Suppress known third-party CUQI import warnings in a local context."""

    with warnings.catch_warnings():
        apply_known_cuqi_warning_filters()
        yield

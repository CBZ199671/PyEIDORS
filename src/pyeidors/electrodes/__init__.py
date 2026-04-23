"""PyEIDORS electrode system module.

Keep package import lightweight so helpers such as
``pyeidors.electrodes.layout`` do not eagerly import pattern generation.
"""

__all__ = ["StimMeasPatternManager"]


def __getattr__(name: str):
    if name == "StimMeasPatternManager":
        from .patterns import StimMeasPatternManager

        globals()["StimMeasPatternManager"] = StimMeasPatternManager
        return StimMeasPatternManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

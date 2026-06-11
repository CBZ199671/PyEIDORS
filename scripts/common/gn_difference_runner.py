"""Compatibility shim: implementation lives in :mod:`pyeidors.realtime.gn_difference_runner`.

Aliasing through ``sys.modules`` keeps every import spelling
(``scripts.common.gn_difference_runner``, ``common.gn_difference_runner``)
bound to the same module object as the packaged implementation, so
monkeypatching in tests affects the real module globals.
"""

import sys

from pyeidors.realtime import gn_difference_runner as _impl

sys.modules[__name__] = _impl

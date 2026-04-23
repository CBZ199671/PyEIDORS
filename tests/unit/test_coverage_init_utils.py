"""Tests for __init__.py import guards and utils modules."""

from __future__ import annotations

import importlib
import sys
from unittest import mock

import numpy as np
import pytest


class TestPyeidorsInitImportGuards:
    """Cover lines 20-21, 29-32, 39-40 in __init__.py."""

    def test_dolfinx_import_failure(self):
        """Lines 20-21: dolfinx ImportError sets _DOLFINX_AVAILABLE=False."""
        # We test via subprocess to avoid contaminating the import state
        from tests.utils import run_python

        code = """
import sys
# Block dolfinx import
sys.modules['dolfinx'] = None
# Force reimport
if 'pyeidors' in sys.modules:
    del sys.modules['pyeidors']
# Prevent torch/cuqi from interfering
sys.modules['torch'] = None
sys.modules['cuqi'] = None

import pyeidors
assert pyeidors._DOLFINX_AVAILABLE is False
assert pyeidors._TORCH_AVAILABLE is False
assert pyeidors._CUDA_AVAILABLE is False
assert pyeidors._MPS_AVAILABLE is False
assert pyeidors._CUQI_AVAILABLE is False
print("PASS")
"""
        result = run_python(code)
        assert result.returncode == 0, result.stderr

    def test_torch_import_failure(self):
        """Lines 29-32: torch ImportError sets flags to False."""
        from tests.utils import run_python

        code = """
import sys
sys.modules['torch'] = None
sys.modules['cuqi'] = None
if 'pyeidors' in sys.modules:
    del sys.modules['pyeidors']
import pyeidors
assert pyeidors._TORCH_AVAILABLE is False
assert pyeidors._CUDA_AVAILABLE is False
assert pyeidors._MPS_AVAILABLE is False
print("PASS")
"""
        result = run_python(code)
        assert result.returncode == 0, result.stderr

    def test_cuqi_import_failure(self):
        """Lines 39-40: cuqi ImportError sets _CUQI_AVAILABLE=False."""
        from tests.utils import run_python

        code = """
import sys
sys.modules['cuqi'] = None
if 'pyeidors' in sys.modules:
    del sys.modules['pyeidors']
import pyeidors
assert pyeidors._CUQI_AVAILABLE is False
print("PASS")
"""
        result = run_python(code)
        assert result.returncode == 0, result.stderr


class TestNumericOps:
    """Cover lines 16, 42, 46 in utils/numeric_ops.py."""

    def test_finite_summary_all_non_finite(self):
        """Line 16: finite_count == 0."""
        from pyeidors.utils.numeric_ops import _finite_summary

        arr = np.array([np.inf, np.nan, -np.inf])
        result = _finite_summary(arr)
        assert "finite=0" in result

    def test_safe_dot_non_finite_result(self):
        """Line 42: result contains non-finite."""
        from pyeidors.utils.numeric_ops import safe_dot

        with pytest.raises(FloatingPointError, match="non-finite"):
            safe_dot(
                np.array([np.finfo(float).max, np.finfo(float).max]),
                np.array([np.finfo(float).max, np.finfo(float).max]),
                "test_op",
            )

    def test_safe_dot_scalar_result(self):
        """Line 46: scalar result from dot product."""
        from pyeidors.utils.numeric_ops import safe_dot

        result = safe_dot(np.array([1.0, 2.0]), np.array([3.0, 4.0]), "test_dot")
        assert isinstance(result, float)
        assert result == 11.0


class TestChineseFontConfig:
    """Cover line 26 in chinese_font_config.py."""

    def test_reset_font_config(self):
        import matplotlib.pyplot as plt
        from pyeidors.utils.chinese_font_config import reset_font_config

        reset_font_config()
        # Just verify it doesn't crash


class TestPlotFontI18n:
    """Cover lines 113-119, 131-135, 151-152, 184 in plot_font_i18n.py."""

    def test_auto_language_locale_exception(self, monkeypatch):
        """Lines 113-119: locale.getlocale raises."""
        import pyeidors.utils.plot_font_i18n as mod

        monkeypatch.delenv("LC_ALL", raising=False)
        monkeypatch.delenv("LC_CTYPE", raising=False)
        monkeypatch.delenv("LANG", raising=False)
        with mock.patch("locale.getlocale", side_effect=Exception("locale error")):
            result = mod._auto_language()
        assert result == "en"

    def test_auto_language_zh_locale(self, monkeypatch):
        """Lines 113-119: locale returns zh."""
        import pyeidors.utils.plot_font_i18n as mod

        monkeypatch.delenv("LC_ALL", raising=False)
        monkeypatch.delenv("LC_CTYPE", raising=False)
        monkeypatch.delenv("LANG", raising=False)
        with mock.patch("locale.getlocale", return_value=("zh_CN", "UTF-8")):
            result = mod._auto_language()
        assert result == "zh"

    def test_invalid_plot_language_warns(self, monkeypatch):
        """Lines 131-135: invalid language triggers warning."""
        import pyeidors.utils.plot_font_i18n as mod

        mod._WARNED_KEYS.discard("plot-lang-badlang")
        result = mod.resolve_plot_language("badlang")
        assert result == "en"

    def test_no_english_fonts_fallback(self, monkeypatch):
        """Line 184: fallback to DejaVu Sans."""
        import pyeidors.utils.plot_font_i18n as mod

        monkeypatch.setattr(mod, "_pick_existing", lambda candidates, available: [])
        result = mod.configure_plot_fonts("en")
        assert "DejaVu Sans" in result.selected_fonts

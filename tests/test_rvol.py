"""Tests for Relative Volume (RVOL)."""

import numpy as np
import pandas as pd

from indikator.rvol import rvol


def test_rvol_basic():
  """Test basic RVOL calculation."""
  # Volume 100, 100... 200.
  # RVOL of last point should be ~2.0 (if window covers 100s).

  volume = pd.Series([100.0] * 19 + [200.0])

  result = rvol(volume, window=20)
  assert hasattr(result, "to_pandas")
  res = result.to_pandas()

  assert res.name == "rvol"

  # SMA of prev 20: (19*100 + 200)/20 = 2100/20 = 105
  # RVOL = 200 / 105 = 1.904...
  # Wait, window includes current? Pandas rolling includes current.

  assert np.isclose(res.iloc[-1], 200.0 / 105.0)


def test_rvol_zero_division():
  """Test division by zero behavior."""
  volume = pd.Series([0.0] * 20)

  result = rvol(volume, window=10)
  res = result.to_pandas()

  # Should be 1.0 (default for zero/low volume) or 0?
  # Implementation sets ones_like as default and only updates if valid_sma.
  # If sma is 0, it stays 1.0.
  assert np.allclose(res, 1.0)

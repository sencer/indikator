"""Tests for Z-Score indicator."""

import numpy as np
import pandas as pd

from indikator.zscore import zscore


def test_zscore_basic():
  """Test basic Z-Score calculation."""
  # Data: 10, 10, 10... 20.
  # Mean close to 10. Std dev small.
  # Z-Score of last point should be large positive.

  data = pd.Series([10.0] * 20 + [20.0])

  result = zscore(data, period=20)
  assert hasattr(result, "to_pandas")
  res = result.to_pandas()
  assert res.name == "zscore"

  # Last value
  assert res.iloc[-1] > 2.0  # More than 2 sigmas


def test_zscore_zero_std():
  """Test behavior when standard deviation is zero."""
  # Constant data -> Z-Score should be 0 (or handled gracefully)
  data = pd.Series([10.0] * 50)

  result = zscore(data, period=10)
  res = result.to_pandas()

  # When std is 0, zscore is typically 0
  assert np.allclose(res.dropna(), 0.0)

"""Tests for Sector Correlation."""

import numpy as np
import pandas as pd

from indikator.sector_correlation import sector_correlation


def test_correlation_basic():
  """Test basic correlation."""
  # Perfectly correlated
  s1 = pd.Series([1, 2, 3, 4, 5])
  s2 = pd.Series([2, 4, 6, 8, 10])

  result = sector_correlation(s1, s2, period=3)
  assert hasattr(result, "to_pandas")
  res = result.to_pandas()

  assert res.name == "correlation"

  # Correlation of 1.0 (after window)
  valid = res.dropna()
  assert np.allclose(valid, 1.0)


def test_correlation_inverse():
  """Test inverse correlation."""
  s1 = pd.Series([1, 2, 3, 4, 5])
  s2 = pd.Series([5, 4, 3, 2, 1])

  result = sector_correlation(s1, s2, period=3)
  res = result.to_pandas()

  valid = res.dropna()
  assert np.allclose(valid, -1.0)


def test_correlation_alignment():
  """Test alignment of indices."""
  idx1 = pd.date_range("2024-01-01", periods=5)
  idx2 = pd.date_range("2024-01-01", periods=5)

  s1 = pd.Series(np.arange(5), index=idx1)
  s2 = pd.Series(np.arange(5), index=idx2)

  # Remove one from s2
  s2 = s2.drop(idx2[2])

  # Correlation should handle missing data (by alignment)
  result = sector_correlation(s1, s2, period=3)
  res = result.to_pandas()

  assert len(res) == 5
  # Where mismatch happens, might be NaN if rolling window affected

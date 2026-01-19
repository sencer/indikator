"""Tests for Churn Factor."""

import numpy as np
import pandas as pd

from indikator.churn_factor import churn_factor


def test_churn_basic():
  """Test basic Churn calculation."""
  # High Vol, Low Range -> High Churn
  # Vol=1000, Range=1. Churn = 1000.

  high = pd.Series([101.0] * 10)
  low = pd.Series([100.0] * 10)  # Range = 1
  volume = pd.Series([1000.0] * 10)

  result = churn_factor(high, low, volume)
  assert hasattr(result, "to_pandas")
  res = result.to_pandas()

  assert res.name == "churn"
  assert np.allclose(res, 1000.0)


def test_churn_zero_range():
  """Test Churn with zero range (division by zero)."""
  high = pd.Series([100.0] * 10)
  low = pd.Series([100.0] * 10)  # Range = 0
  volume = pd.Series([1000.0] * 10)

  result = churn_factor(high, low, volume)
  res = result.to_pandas()

  # Should be 0.0 (or handled)
  assert np.allclose(res, 0.0)

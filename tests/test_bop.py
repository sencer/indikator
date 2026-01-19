"""Tests for BOP (Balance of Power) indicator."""

import numpy as np
import pandas as pd
import pytest

from indikator.bop import bop

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


def test_bop_basic():
  """Test basic BOP calculation."""
  # BOP = (Close - Open) / (High - Low)
  open_ = pd.Series([10.0, 10.0, 10.0, 10.0])
  high = pd.Series([12.0, 15.0, 12.0, 12.0])
  low = pd.Series([8.0, 5.0, 8.0, 8.0])
  close = pd.Series([11.0, 10.0, 9.0, 12.0])

  # 1: (11-10)/(12-8) = 1/4 = 0.25
  # 2: (10-10)/(15-5) = 0/10 = 0.0
  # 3: (9-10)/(12-8) = -1/4 = -0.25
  # 4: (12-10)/(12-8) = 2/4 = 0.5

  result = bop(open_, high, low, close).to_pandas()
  # result is pd.Series directly

  assert len(result) == 4
  assert np.isclose(result[0], 0.25)
  assert np.isclose(result[1], 0.0)
  assert np.isclose(result[2], -0.25)
  assert np.isclose(result[3], 0.5)


def test_bop_zero_range():
  """Test BOP when High == Low."""
  # Should be 0.0
  open_ = pd.Series([10.0])
  high = pd.Series([10.0])
  low = pd.Series([10.0])
  close = pd.Series([10.0])

  result = bop(open_, high, low, close).to_pandas()
  assert result[0] == 0.0


@pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
def test_bop_matches_talib():
  """Test BOP matches TA-Lib."""
  np.random.seed(42)
  n = 100
  open_ = pd.Series(np.random.uniform(10, 20, n))
  high = pd.Series(np.random.uniform(20, 30, n))
  low = pd.Series(np.random.uniform(5, 10, n))
  close = pd.Series(np.random.uniform(10, 25, n))

  # Ensure High >= Low
  high = np.maximum(high, low)

  result = bop(open_, high, low, close).to_pandas()
  expected = talib.BOP(open_.values, high.values, low.values, close.values)

  np.testing.assert_allclose(result.values, expected, rtol=1e-10)

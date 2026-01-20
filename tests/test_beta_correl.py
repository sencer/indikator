"""Tests for BETA and CORREL statistical indicators."""

import numpy as np
import pandas as pd
import pytest

from indikator.beta import beta, beta_statistical
from indikator.correl import correl

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


def test_beta_statistical_basic():
  """Test basic BETA_STATISTICAL calculation (raw inputs)."""
  np.random.seed(42)
  # Market returns (x)
  market = pd.Series(np.random.randn(100) * 0.01)  # 1% std
  # Stock returns with beta ~2
  stock = market * 2 + np.random.randn(100) * 0.005

  result = beta_statistical(market, stock, period=30)
  res = result.to_pandas()

  assert res.name == "beta"
  assert len(res) == 100
  # First 29 should be NaN
  assert res[:29].isna().all()
  # Beta should be approximately 2 (will vary due to noise)
  assert 1.0 < res.dropna().mean() < 3.0


def test_beta_basic():
  """Test new BETA calculation (implicit returns)."""
  np.random.seed(42)
  # Using cumulative sum to create price series
  x = pd.Series(np.cumsum(np.random.randn(100)) + 100)
  y = pd.Series(np.cumsum(np.random.randn(100)) + 100)

  result = beta(x, y, period=30)
  res = result.to_pandas()

  assert res.name == "beta"
  assert len(res) == 100
  # First 30 should be NaN (1 for ROCP + 29 for Beta window)
  assert res[:30].isna().all()


def test_correl_basic():
  """Test basic CORREL calculation."""
  np.random.seed(42)
  x = pd.Series(np.random.randn(100))
  # Y is correlated with X
  y = x * 0.8 + np.random.randn(100) * 0.2

  result = correl(x, y, period=30)
  res = result.to_pandas()

  assert res.name == "correl"
  assert len(res) == 100
  # First 29 should be NaN
  assert res[:29].isna().all()
  # Correlation should be high (close to 1)
  assert res.dropna().mean() > 0.5


def test_correl_perfect_correlation():
  """Test CORREL with perfectly correlated data."""
  x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
  y = x * 2 + 1  # Perfect linear relationship

  result = correl(x, y, period=5)
  res = result.to_pandas()

  # Should be 1.0 for perfect correlation
  assert np.isclose(res.iloc[-1], 1.0)


def test_beta_insufficient_data():
  """Test BETA with insufficient data."""
  x = pd.Series([1.0, 2.0, 3.0])
  y = pd.Series([1.0, 2.0, 3.0])
  result = beta(x, y, period=10)
  res = result.to_pandas()
  assert res.isna().all()


@pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
def test_beta_matches_talib_direct():
  """Test new BETA matches TA-Lib directly (both do implicit returns)."""
  np.random.seed(42)
  x = pd.Series(np.cumsum(np.random.randn(100)) + 100)
  y = pd.Series(np.cumsum(np.random.randn(100)) + 100)
  period = 5

  # Now our beta() implementation matches TA-Lib's behavior
  # We should match output exactly without manual transformations
  result = beta(x, y, period=period).to_pandas()
  expected = talib.BETA(x.values, y.values, timeperiod=period)

  pd.testing.assert_series_equal(
    result, pd.Series(expected, index=x.index, name="beta")
  )


@pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
def test_beta_statistical_matches_manual():
  """Test BETA_STATISTICAL matches manual calculation."""
  # This tests the raw beta calculation logic
  x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
  y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # y = 2x, exact beta=2
  period = 5

  # beta_statistical takes raw inputs
  result = beta_statistical(pd.Series(x), pd.Series(y), period=period).to_pandas()

  # Should be exactly 2.0 for the last point
  assert np.isclose(result.iloc[-1], 2.0)


@pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
def test_correl_matches_talib():
  """Test CORREL matches TA-Lib."""
  np.random.seed(42)
  x = pd.Series(np.random.randn(100) + 100)
  y = pd.Series(np.random.randn(100) + 100)
  period = 30

  result = correl(x, y, period=period).to_pandas()
  expected = talib.CORREL(x.values, y.values, timeperiod=period)
  pd.testing.assert_series_equal(
    result, pd.Series(expected, index=x.index, name="correl")
  )

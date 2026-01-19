"""Tests for price transform indicators."""

import numpy as np
import pandas as pd
import pytest

from indikator.avgprice import avgprice
from indikator.medprice import medprice
from indikator.typprice import typprice
from indikator.wclprice import wclprice

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


@pytest.fixture
def ohlc_data():
  """Generate OHLC test data."""
  np.random.seed(42)
  n = 100
  close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
  high = close + np.abs(np.random.randn(n) * 0.3)
  low = close - np.abs(np.random.randn(n) * 0.3)
  open_ = close + np.random.randn(n) * 0.2
  return {
    "open": pd.Series(open_),
    "high": pd.Series(high),
    "low": pd.Series(low),
    "close": pd.Series(close),
  }


class TestTypprice:
  """Tests for TYPPRICE indicator."""

  def test_typprice_basic(self, ohlc_data):
    """Test TYPPRICE calculation."""
    result = typprice(
      ohlc_data["high"], ohlc_data["low"], ohlc_data["close"]
    ).to_pandas()

    assert len(result) == len(ohlc_data["high"])
    assert not result.isna().any()

  def test_typprice_formula(self):
    """Verify TYPPRICE formula."""
    high = pd.Series([110.0, 120.0, 130.0])
    low = pd.Series([90.0, 100.0, 110.0])
    close = pd.Series([100.0, 110.0, 120.0])

    result = typprice(high, low, close).to_pandas()

    expected = (high + low + close) / 3
    pd.testing.assert_series_equal(result, expected.rename("typprice"))

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_typprice_matches_talib(self, ohlc_data):
    """Test TYPPRICE matches TA-Lib."""
    result = typprice(
      ohlc_data["high"], ohlc_data["low"], ohlc_data["close"]
    ).to_pandas()
    expected = talib.TYPPRICE(
      ohlc_data["high"].values,
      ohlc_data["low"].values,
      ohlc_data["close"].values,
    )

    np.testing.assert_allclose(result.values, expected, rtol=1e-10)


class TestMedprice:
  """Tests for MEDPRICE indicator."""

  def test_medprice_basic(self, ohlc_data):
    """Test MEDPRICE calculation."""
    result = medprice(ohlc_data["high"], ohlc_data["low"]).to_pandas()

    assert len(result) == len(ohlc_data["high"])
    assert not result.isna().any()

  def test_medprice_formula(self):
    """Verify MEDPRICE formula."""
    high = pd.Series([110.0, 120.0, 130.0])
    low = pd.Series([90.0, 100.0, 110.0])

    result = medprice(high, low).to_pandas()

    expected = (high + low) / 2
    pd.testing.assert_series_equal(result, expected.rename("medprice"))

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_medprice_matches_talib(self, ohlc_data):
    """Test MEDPRICE matches TA-Lib."""
    result = medprice(ohlc_data["high"], ohlc_data["low"]).to_pandas()
    expected = talib.MEDPRICE(ohlc_data["high"].values, ohlc_data["low"].values)

    np.testing.assert_allclose(result.values, expected, rtol=1e-10)


class TestWclprice:
  """Tests for WCLPRICE indicator."""

  def test_wclprice_basic(self, ohlc_data):
    """Test WCLPRICE calculation."""
    result = wclprice(
      ohlc_data["high"], ohlc_data["low"], ohlc_data["close"]
    ).to_pandas()

    assert len(result) == len(ohlc_data["high"])
    assert not result.isna().any()

  def test_wclprice_formula(self):
    """Verify WCLPRICE formula."""
    high = pd.Series([110.0, 120.0, 130.0])
    low = pd.Series([90.0, 100.0, 110.0])
    close = pd.Series([100.0, 110.0, 120.0])

    result = wclprice(high, low, close).to_pandas()

    expected = (high + low + 2 * close) / 4
    pd.testing.assert_series_equal(result, expected.rename("wclprice"))

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_wclprice_matches_talib(self, ohlc_data):
    """Test WCLPRICE matches TA-Lib."""
    result = wclprice(
      ohlc_data["high"], ohlc_data["low"], ohlc_data["close"]
    ).to_pandas()
    expected = talib.WCLPRICE(
      ohlc_data["high"].values,
      ohlc_data["low"].values,
      ohlc_data["close"].values,
    )

    np.testing.assert_allclose(result.values, expected, rtol=1e-10)


class TestAvgprice:
  """Tests for AVGPRICE indicator."""

  def test_avgprice_basic(self, ohlc_data):
    """Test AVGPRICE calculation."""
    result = avgprice(
      ohlc_data["open"], ohlc_data["high"], ohlc_data["low"], ohlc_data["close"]
    ).to_pandas()

    assert len(result) == len(ohlc_data["high"])
    assert not result.isna().any()

  def test_avgprice_formula(self):
    """Verify AVGPRICE formula."""
    open_ = pd.Series([95.0, 105.0, 115.0])
    high = pd.Series([110.0, 120.0, 130.0])
    low = pd.Series([90.0, 100.0, 110.0])
    close = pd.Series([100.0, 110.0, 120.0])

    result = avgprice(open_, high, low, close).to_pandas()

    expected = (open_ + high + low + close) / 4
    pd.testing.assert_series_equal(result, expected.rename("avgprice"))

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_avgprice_matches_talib(self, ohlc_data):
    """Test AVGPRICE matches TA-Lib."""
    result = avgprice(
      ohlc_data["open"], ohlc_data["high"], ohlc_data["low"], ohlc_data["close"]
    ).to_pandas()
    expected = talib.AVGPRICE(
      ohlc_data["open"].values,
      ohlc_data["high"].values,
      ohlc_data["low"].values,
      ohlc_data["close"].values,
    )

    np.testing.assert_allclose(result.values, expected, rtol=1e-10)

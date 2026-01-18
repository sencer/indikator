"""Tests for MFI (Money Flow Index) indicator."""

import numpy as np
import pandas as pd
import pytest

from indikator.mfi import mfi

# Try to import talib for comparison tests
try:
  import talib  # type: ignore[import-untyped]

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


class TestMFI:
  """Tests for MFI indicator."""

  def test_mfi_basic(self):
    """Test MFI basic calculation."""
    data = pd.DataFrame({
      "high": [
        102.0,
        104.0,
        103.0,
        106.0,
        108.0,
        107.0,
        109.0,
        108.0,
        110.0,
        112.0,
      ]
      * 2,
      "low": [
        100.0,
        101.0,
        100.0,
        103.0,
        105.0,
        104.0,
        106.0,
        105.0,
        107.0,
        109.0,
      ]
      * 2,
      "close": [
        101.0,
        103.0,
        102.0,
        105.0,
        107.0,
        106.0,
        108.0,
        107.0,
        109.0,
        111.0,
      ]
      * 2,
      "volume": [
        1000.0,
        1200.0,
        900.0,
        1500.0,
        1100.0,
        1300.0,
        1000.0,
        1400.0,
        1200.0,
        1100.0,
      ]
      * 2,
    })

    result = mfi(data, window=5)

    # Check return type is Series
    assert isinstance(result, pd.Series)
    assert result.name == "mfi"

    # Check shape
    assert len(result) == len(data)

    # Check MFI is calculated after window
    assert result.isna().iloc[:5].all()
    assert not result.isna().iloc[5:].all()

    # Check MFI is in range [0, 100]
    assert (result.dropna() >= 0).all()
    assert (result.dropna() <= 100).all()

  def test_mfi_typical_price(self):
    """Test MFI typical price calculation."""
    data = pd.DataFrame({
      "high": [102.0, 104.0],
      "low": [100.0, 102.0],
      "close": [101.0, 103.0],
      "volume": [1000.0, 1200.0],
    })

    result = mfi(data, window=2)

    # Now returns only MFI values (typical_price is internal)
    assert isinstance(result, pd.Series)
    assert result.name == "mfi"
    assert len(result) == len(data)

  def test_mfi_empty_data(self):
    """Test MFI with empty dataframe."""
    data = pd.DataFrame(columns=["high", "low", "close", "volume"]).astype(float)

    with pytest.raises(ValueError, match="not empty"):
      mfi(data)

  def test_mfi_validation_missing_columns(self):
    """Test MFI validation with missing columns."""
    data = pd.DataFrame({
      "high": [102.0, 104.0],
      "low": [100.0, 102.0],
      "close": [101.0, 103.0],
    })

    with pytest.raises((ValueError, KeyError)):
      mfi(data)

  def test_mfi_window_parameter(self):
    """Test MFI with different window sizes."""
    data = pd.DataFrame({
      "high": [102.0, 104.0, 103.0, 106.0, 108.0] * 4,
      "low": [100.0, 101.0, 100.0, 103.0, 105.0] * 4,
      "close": [101.0, 103.0, 102.0, 105.0, 107.0] * 4,
      "volume": [1000.0, 1200.0, 900.0, 1500.0, 1100.0] * 4,
    })

    result_short = mfi(data, window=3)
    result_long = mfi(data, window=10)

    # Short window should have values earlier
    assert result_short.notna().sum() > result_long.notna().sum()

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_mfi_matches_talib(self):
    """Test MFI matches TA-Lib output."""
    np.random.seed(42)
    n = 100
    close = pd.Series(100.0 + np.cumsum(np.random.randn(n) * 0.5))
    high = close + np.random.rand(n) * 2
    low = close - np.random.rand(n) * 2
    volume = pd.Series(1000.0 + np.random.rand(n) * 500)

    data = pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume})

    result = mfi(data, window=14)
    expected = pd.Series(
      talib.MFI(high.values, low.values, close.values, volume.values, timeperiod=14),
    )

    # Compare non-NaN values
    valid_mask = result.notna() & expected.notna()
    np.testing.assert_allclose(
      result[valid_mask].values,
      expected[valid_mask].values,
      rtol=1e-10,
    )

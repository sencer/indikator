"""Tests for RSI (Relative Strength Index) indicator."""

from datawarden.exceptions import ValidationError
import numpy as np
import pandas as pd
import pytest

from indikator.rsi import rsi

# Try to import talib for comparison tests
try:
  import talib  # type: ignore[import-untyped]

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


class TestRSI:
  """Tests for RSI indicator."""

  def test_rsi_basic(self):
    """Test RSI basic calculation."""
    prices = pd.Series(
      [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0] * 2,
    )

    result = rsi(prices, window=5).to_pandas()

    # Check shape
    assert len(result) == len(prices)

    # Check RSI is calculated after window
    assert result.isna().iloc[:5].all()  # First 5 bars should be NaN
    assert not result.isna().iloc[5:].all()  # After window should have values

    # Check RSI is in range [0, 100]
    assert (result.dropna() >= 0).all()
    assert (result.dropna() <= 100).all()

  def test_rsi_extreme_values(self):
    """Test RSI with extreme price movements."""
    # All gains
    prices_up = pd.Series([
      100.0,
      101.0,
      102.0,
      103.0,
      104.0,
      105.0,
      106.0,
      107.0,
      108.0,
      109.0,
    ])
    result_up = rsi(prices_up, window=5).to_pandas()

    # RSI should be high (close to 100) for all gains
    assert (result_up.dropna() > 80).all()

    # All losses
    prices_down = pd.Series([
      109.0,
      108.0,
      107.0,
      106.0,
      105.0,
      104.0,
      103.0,
      102.0,
      101.0,
      100.0,
    ])
    result_down = rsi(prices_down, window=5).to_pandas()

    # RSI should be low (close to 0) for all losses
    assert (result_down.dropna() < 20).all()

  def test_rsi_empty_data(self) -> None:
    """Should raise ValueError when data is empty."""
    empty_data = pd.Series([], dtype=float)
    with pytest.raises((ValueError, ValidationError), match="empty"):
      rsi(empty_data)

  def test_rsi_insufficient_data(self):
    """Test RSI with insufficient data."""
    prices = pd.Series([100.0, 101.0])
    result = rsi(prices, window=5).to_pandas()

    # Should return all NaN
    assert result.isna().all()

  def test_rsi_window_parameter(self):
    """Test RSI with different window sizes."""
    prices = pd.Series(
      [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0] * 3,
    )

    result_short = rsi(prices, window=3).to_pandas()
    result_long = rsi(prices, window=10).to_pandas()

    # Short window should have values earlier
    assert result_short.notna().sum() > result_long.notna().sum()

  def test_rsi_with_inf(self):
    """Test RSI with Inf values."""
    prices = pd.Series([100.0, 102.0, np.inf, 103.0, 105.0] * 5)

    with pytest.raises((ValueError, ValidationError), match="Finite"):
      rsi(prices)

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_rsi_matches_talib(self):
    """Test RSI matches TA-Lib output."""
    np.random.seed(42)
    prices = pd.Series(100.0 + np.cumsum(np.random.randn(100) * 0.5))

    result = rsi(prices, window=14).to_pandas()
    expected = pd.Series(talib.RSI(prices.values, timeperiod=14))

    # Compare non-NaN values (TA-Lib may have different NaN handling)
    valid_mask = result.notna() & expected.notna()
    np.testing.assert_allclose(
      result[valid_mask].values,
      expected[valid_mask].values,
      rtol=1e-10,
    )

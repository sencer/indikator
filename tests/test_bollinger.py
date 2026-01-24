"""Tests for Bollinger Bands indicator."""

from datawarden.exceptions import ValidationError
import numpy as np
import pandas as pd
import pytest

from indikator._results import BollingerResult
from indikator.bollinger import bollinger_bands, bollinger_with_bandwidth


# Try to import talib for comparison tests
try:
  import talib  # type: ignore[import-untyped]

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


class TestBollingerBands:
  """Tests for Bollinger Bands indicator."""

  def test_bollinger_basic(self):
    """Test Bollinger Bands basic calculation."""
    prices = pd.Series(
      [
        100.0,
        102.0,
        101.0,
        103.0,
        105.0,
        104.0,
        106.0,
        108.0,
        107.0,
        109.0,
        111.0,
        110.0,
      ]
      * 2,
    )

    res_obj = bollinger_with_bandwidth(prices, window=5, num_std=2.0)
    assert isinstance(res_obj, BollingerResult)

    result = res_obj.to_pandas()

    # Check columns
    assert "bb_middle" in result.columns
    assert "bb_upper" in result.columns
    assert "bb_lower" in result.columns
    assert "bb_bandwidth" in result.columns
    assert "bb_percent" in result.columns

    # Check shape
    assert len(result) == len(prices)

    # Check middle band is SMA
    expected_middle = prices.rolling(window=5).mean()
    pd.testing.assert_series_equal(
      result["bb_middle"],
      expected_middle,
      check_names=False,
    )

    # Check upper band >= middle band (where both are not NaN)
    valid_mask = result["bb_upper"].notna() & result["bb_middle"].notna()
    assert (result["bb_upper"][valid_mask] >= result["bb_middle"][valid_mask]).all()

    # Check lower band <= middle band (where both are not NaN)
    assert (result["bb_lower"][valid_mask] <= result["bb_middle"][valid_mask]).all()

  def test_bollinger_dataframe(self):
    """Test Bollinger Bands with DataFrame input."""
    data = pd.DataFrame({
      "close": [100.0, 102.0, 101.0, 103.0, 105.0],
      "volume": [100, 200, 150, 180, 220],
    })
    # Pass Series directly
    res_obj = bollinger_with_bandwidth(data["close"], window=5, num_std=2.0)
    assert isinstance(res_obj, BollingerResult)

    result = res_obj.to_pandas()
    assert isinstance(result, pd.DataFrame)
    assert "bb_middle" in result.columns

  def test_bollinger_percent_b(self):
    """Test Bollinger %B calculation."""
    prices = pd.Series(
      [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0] * 2,
    )

    result_obj = bollinger_with_bandwidth(prices, window=5, num_std=2.0)
    # Check directly on object if convenient, or convert
    # NamedTuple access
    assert result_obj.bb_percent is not None

    result = result_obj.to_pandas()
    assert result["bb_percent"].notna().any()

  def test_bollinger_empty_data(self) -> None:
    """Should raise ValueError when data is empty."""
    empty_data = pd.Series([], dtype=float)
    with pytest.raises((ValueError, ValidationError), match="empty"):
      bollinger_bands(empty_data)

  def test_bollinger_window_parameter(self):
    """Test Bollinger Bands with different window sizes."""
    prices = pd.Series(
      [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0] * 3,
    )

    result_short = bollinger_with_bandwidth(prices, window=3, num_std=2.0).to_pandas()
    result_long = bollinger_with_bandwidth(prices, window=15, num_std=2.0).to_pandas()

    # Different windows should produce different results
    assert not result_short["bb_middle"].equals(result_long["bb_middle"])

  def test_bollinger_invalid_input(self):
    """Test Bollinger Bands with invalid input."""
    # Infinite values
    data = pd.Series([100.0, np.inf, 102.0])
    with pytest.raises((ValueError, ValidationError), match="Finite"):
      bollinger_bands(data)

  def test_bollinger_with_inf(self):
    """Test Bollinger Bands with Inf values."""
    prices = pd.Series([100.0, 102.0, np.inf, 103.0, 105.0])

    with pytest.raises((ValueError, ValidationError), match="Finite"):
      bollinger_bands(prices)

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_bollinger_matches_talib(self):
    """Test Bollinger Bands matches TA-Lib (population std)."""
    np.random.seed(42)
    prices = pd.Series(100.0 + np.cumsum(np.random.randn(100) * 0.5))

    # bollinger_bands() uses ddof=0 which matches TA-Lib
    res_obj = bollinger_bands(prices, window=20, num_std=2.0).to_pandas()
    exp_u, exp_m, exp_l = talib.BBANDS(
      prices.values, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0
    )

    valid_mask = res_obj["bb_middle"].notna() & np.isfinite(exp_m)

    # Compare middle
    np.testing.assert_allclose(
      res_obj["bb_middle"][valid_mask].values, exp_m[valid_mask], rtol=1e-10
    )
    # Compare upper
    np.testing.assert_allclose(
      res_obj["bb_upper"][valid_mask].values, exp_u[valid_mask], rtol=1e-10
    )
    # Compare lower
    np.testing.assert_allclose(
      res_obj["bb_lower"][valid_mask].values, exp_l[valid_mask], rtol=1e-10
    )

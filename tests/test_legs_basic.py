"""Basic functionality tests for zigzag_legs indicator."""

from __future__ import annotations

import pandas as pd
import pytest

from indikator.legs import zigzag_legs


class TestBasicFunctionality:
  """Test core functionality of zigzag_legs."""

  def test_returns_series_named_zigzag_legs(
    self, simple_uptrend_df: pd.DataFrame
  ) -> None:
    """Should return Series named zigzag_legs."""
    result = zigzag_legs(simple_uptrend_df["close"])

    assert isinstance(result, pd.Series)
    assert result.name == "zigzag_legs"

  def test_does_not_modify_original_series(
    self, simple_uptrend_df: pd.DataFrame
  ) -> None:
    """Should not modify the input Series."""
    series = simple_uptrend_df["close"].copy()
    _ = zigzag_legs(series)

    pd.testing.assert_series_equal(series, simple_uptrend_df["close"])

  def test_output_length_matches_input(self, simple_uptrend_df: pd.DataFrame) -> None:
    """Output should have same length as input."""
    result = zigzag_legs(simple_uptrend_df["close"])
    assert len(result) == len(simple_uptrend_df)

  def test_empty_series_returns_empty_series(
    self,
    empty_df: pd.DataFrame,  # noqa: ARG002
  ) -> None:
    """Should raise ValueError if input Series is empty."""
    with pytest.raises(ValueError, match="Data must not be empty"):
      zigzag_legs(pd.Series(dtype=float))

  def test_single_value_returns_zero_legs(self, single_value_df: pd.DataFrame) -> None:
    """Single price should result in zero legs (no trend established)."""
    result = zigzag_legs(single_value_df["close"])

    assert result.iloc[0] == 0.0

  def test_flat_prices_return_zero_legs(self, flat_prices_df: pd.DataFrame) -> None:
    """Constant prices should not establish any trend."""
    result = zigzag_legs(flat_prices_df["close"], threshold=0.01)

    # All values should be 0 since there's no price movement
    assert (result == 0.0).all()


class TestSimpleTrends:
  """Test basic trend detection."""

  def test_simple_uptrend_creates_positive_legs(
    self, simple_uptrend_df: pd.DataFrame
  ) -> None:
    """Uptrend should create positive leg counts."""
    result = zigzag_legs(
      simple_uptrend_df["close"], threshold=0.01, confirmation_bars=0
    )

    # After threshold is crossed, we should see positive values
    legs = result.values

    # First few bars might be 0 until threshold is crossed
    # Once trend is established, legs should be positive
    assert legs[-1] > 0  # Final value should be positive
    assert (legs >= 0).all()  # No negative values in pure uptrend

  def test_simple_downtrend_creates_negative_legs(
    self, simple_downtrend_df: pd.DataFrame
  ) -> None:
    """Downtrend should create negative leg counts."""
    result = zigzag_legs(
      simple_downtrend_df["close"], threshold=0.01, confirmation_bars=0
    )

    legs = result.values

    # Final value should be negative in downtrend
    assert legs[-1] < 0
    assert (legs <= 0).all()  # No positive values in pure downtrend

  def test_threshold_parameter_affects_trend_detection(
    self, simple_uptrend_df: pd.DataFrame
  ) -> None:
    """Higher threshold should delay trend detection."""
    # Low threshold - should detect trend quickly
    result_low = zigzag_legs(simple_uptrend_df["close"], threshold=0.01)

    # High threshold - might not detect trend at all
    result_high = zigzag_legs(simple_uptrend_df["close"], threshold=0.50)

    # Low threshold should have more non-zero values
    non_zero_low = (result_low != 0).sum()
    non_zero_high = (result_high != 0).sum()

    assert non_zero_low >= non_zero_high


class TestSignedOutput:
  """Test that output correctly uses signed values."""

  def test_uptrend_has_positive_sign(self, simple_uptrend_df: pd.DataFrame) -> None:
    """Bullish trends should have positive leg counts."""
    result = zigzag_legs(simple_uptrend_df["close"], threshold=0.01)

    non_zero = result[result != 0]
    if len(non_zero) > 0:
      assert (non_zero > 0).all()

  def test_downtrend_has_negative_sign(self, simple_downtrend_df: pd.DataFrame) -> None:
    """Bearish trends should have negative leg counts."""
    result = zigzag_legs(simple_downtrend_df["close"], threshold=0.01)

    non_zero = result[result != 0]
    if len(non_zero) > 0:
      assert (non_zero < 0).all()

  def test_zero_indicates_no_trend(self, flat_prices_df: pd.DataFrame) -> None:
    """Zero values indicate no established trend."""
    result = zigzag_legs(flat_prices_df["close"], threshold=0.01)

    assert (result == 0.0).all()


class TestParameterValidation:
  """Test parameter validation and constraints."""

  def test_default_parameters_work(self, simple_uptrend_df: pd.DataFrame) -> None:
    """Should work with all default parameters."""
    result = zigzag_legs(simple_uptrend_df["close"])
    assert isinstance(result, pd.Series)

  def test_custom_threshold(self, simple_uptrend_df: pd.DataFrame) -> None:
    """Should accept custom threshold parameter."""
    result = zigzag_legs(simple_uptrend_df["close"], threshold=0.05)
    assert isinstance(result, pd.Series)

  def test_custom_min_distance(self, simple_uptrend_df: pd.DataFrame) -> None:
    """Should accept custom min_distance_pct parameter."""
    result = zigzag_legs(simple_uptrend_df["close"], min_distance_pct=0.01)
    assert isinstance(result, pd.Series)

  def test_custom_confirmation_bars(self, simple_uptrend_df: pd.DataFrame) -> None:
    """Should accept custom confirmation_bars parameter."""
    result = zigzag_legs(simple_uptrend_df["close"], confirmation_bars=5)
    assert isinstance(result, pd.Series)

  def test_zero_confirmation_bars(self, simple_uptrend_df: pd.DataFrame) -> None:
    """Should work with zero confirmation (immediate reversals)."""
    result = zigzag_legs(simple_uptrend_df["close"], confirmation_bars=0)
    assert isinstance(result, pd.Series)

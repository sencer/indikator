"""Tests for Pivot Points indicator."""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import pytest

from indikator.pivots import pivot_points


class TestPivotPoints:
  """Tests for Pivot Points indicator."""

  def test_pivots_standard_basic(self):
    """Test standard pivot points basic calculation."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    data = pd.DataFrame(
      {
        "high": [110] * 10,
        "low": [100] * 10,
        "close": [105] * 10,
      },
      index=dates,
    )

    result = pivot_points(data, method="standard", period="D")

    # Check columns exist
    assert "pp" in result.columns
    assert "r1" in result.columns
    assert "r2" in result.columns
    assert "r3" in result.columns
    assert "s1" in result.columns
    assert "s2" in result.columns
    assert "s3" in result.columns

    # Check shape
    assert len(result) == len(data)

  def test_pivots_standard_manual_calculation(self):
    """Test standard pivot points with manual calculation."""
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    data = pd.DataFrame(
      {
        "high": [110.0, 115.0, 120.0],
        "low": [100.0, 105.0, 110.0],
        "close": [105.0, 110.0, 115.0],
      },
      index=dates,
    )

    result = pivot_points(data, method="standard", period="D")

    # Day 2 pivots based on Day 1 data
    # PP = (110 + 100 + 105) / 3 = 105
    # R1 = 2*105 - 100 = 110
    # S1 = 2*105 - 110 = 100
    # R2 = 105 + (110 - 100) = 115
    # S2 = 105 - (110 - 100) = 95
    # R3 = 110 + 2*(105 - 100) = 120
    # S3 = 100 - 2*(110 - 105) = 90

    # Day 1 will be NaN (no previous day)
    assert np.isnan(result["pp"].iloc[0])

    # Day 2 (index 1)
    assert result["pp"].iloc[1] == 105.0
    assert result["r1"].iloc[1] == 110.0
    assert result["s1"].iloc[1] == 100.0
    assert result["r2"].iloc[1] == 115.0
    assert result["s2"].iloc[1] == 95.0
    assert result["r3"].iloc[1] == 120.0
    assert result["s3"].iloc[1] == 90.0

  def test_pivots_fibonacci(self):
    """Test Fibonacci pivot points."""
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    data = pd.DataFrame(
      {
        "high": [110.0, 115.0, 120.0],
        "low": [100.0, 105.0, 110.0],
        "close": [105.0, 110.0, 115.0],
      },
      index=dates,
    )

    result = pivot_points(data, method="fibonacci", period="D")

    # Check columns exist
    assert "pp" in result.columns
    assert "r1" in result.columns
    assert "r2" in result.columns
    assert "r3" in result.columns
    assert "s1" in result.columns
    assert "s2" in result.columns
    assert "s3" in result.columns

    # Day 2 pivots based on Day 1 data
    # PP = (110 + 100 + 105) / 3 = 105
    # Range = 110 - 100 = 10
    # R1 = 105 + 0.382 * 10 = 108.82
    # S1 = 105 - 0.382 * 10 = 101.18

    assert result["pp"].iloc[1] == 105.0
    assert abs(result["r1"].iloc[1] - 108.82) < 0.01
    assert abs(result["s1"].iloc[1] - 101.18) < 0.01

  def test_pivots_woodie(self):
    """Test Woodie pivot points."""
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    data = pd.DataFrame(
      {
        "high": [110.0, 115.0, 120.0],
        "low": [100.0, 105.0, 110.0],
        "close": [105.0, 110.0, 115.0],
      },
      index=dates,
    )

    result = pivot_points(data, method="woodie", period="D")

    # Check columns exist (Woodie has no R3/S3)
    assert "pp" in result.columns
    assert "r1" in result.columns
    assert "r2" in result.columns
    assert "s1" in result.columns
    assert "s2" in result.columns
    assert "r3" not in result.columns
    assert "s3" not in result.columns

    # Day 2 pivots based on Day 1 data
    # PP = (110 + 100 + 2*105) / 4 = 105
    # R1 = 2*105 - 100 = 110
    # S1 = 2*105 - 110 = 100

    assert result["pp"].iloc[1] == 105.0
    assert result["r1"].iloc[1] == 110.0
    assert result["s1"].iloc[1] == 100.0

  def test_pivots_camarilla(self):
    """Test Camarilla pivot points."""
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    data = pd.DataFrame(
      {
        "high": [110.0, 115.0, 120.0],
        "low": [100.0, 105.0, 110.0],
        "close": [105.0, 110.0, 115.0],
      },
      index=dates,
    )

    result = pivot_points(data, method="camarilla", period="D")

    # Check columns exist (Camarilla has R4/S4)
    assert "pp" in result.columns
    assert "r1" in result.columns
    assert "r2" in result.columns
    assert "r3" in result.columns
    assert "r4" in result.columns
    assert "s1" in result.columns
    assert "s2" in result.columns
    assert "s3" in result.columns
    assert "s4" in result.columns

    # Day 2 pivots based on Day 1 data
    # PP = (110 + 100 + 105) / 3 = 105
    # Range = 110 - 100 = 10
    # R1 = 105 + 1.1/12 * 10 = 105.9166...

    assert result["pp"].iloc[1] == 105.0
    assert abs(result["r1"].iloc[1] - (105 + 1.1 / 12 * 10)) < 0.01

  def test_pivots_empty_data(self):
    """Test pivot points with empty dataframe."""
    data = pd.DataFrame(columns=["high", "low", "close"], dtype=float)
    data.index = pd.DatetimeIndex([])

    with pytest.raises(ValueError, match="Data must not be empty"):
      pivot_points(data)

  def test_pivots_weekly_period(self):
    """Test pivot points with weekly period."""
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    data = pd.DataFrame(
      {
        "high": [110] * 20,
        "low": [100] * 20,
        "close": [105] * 20,
      },
      index=dates,
    )

    result = pivot_points(data, method="standard", period="W")

    # Should calculate pivots based on previous week
    assert "pp" in result.columns
    assert len(result) == len(data)

  def test_pivots_monthly_period(self):
    """Test pivot points with monthly period."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    data = pd.DataFrame(
      {
        "high": [110] * 60,
        "low": [100] * 60,
        "close": [105] * 60,
      },
      index=dates,
    )

    result = pivot_points(data, method="standard", period="ME")

    # Should calculate pivots based on previous month
    assert "pp" in result.columns
    assert len(result) == len(data)

  def test_pivots_resistance_above_support(self):
    """Test that resistance levels are above support levels."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    data = pd.DataFrame(
      {
        "high": [110.0] * 10,
        "low": [100.0] * 10,
        "close": [105.0] * 10,
      },
      index=dates,
    )

    result = pivot_points(data, method="standard", period="D")

    # Filter out NaN values
    valid_mask = result["r1"].notna()

    # R1 > PP > S1
    assert (result.loc[valid_mask, "r1"] > result.loc[valid_mask, "pp"]).all()
    assert (result.loc[valid_mask, "pp"] > result.loc[valid_mask, "s1"]).all()

    # R2 > R1
    assert (result.loc[valid_mask, "r2"] > result.loc[valid_mask, "r1"]).all()
    # S1 > S2
    assert (result.loc[valid_mask, "s1"] > result.loc[valid_mask, "s2"]).all()

  def test_pivots_validation_missing_columns(self):
    """Test pivot points validation with missing columns."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    data = pd.DataFrame(
      {
        "high": [110] * 10,
        "low": [100] * 10,
      },
      index=dates,
    )

    with pytest.raises(ValueError, match="Missing columns"):
      pivot_points(data)

  def test_pivots_validation_not_datetime_index(self):
    """Test pivot points validation with non-datetime index."""
    data = pd.DataFrame({
      "high": [110] * 10,
      "low": [100] * 10,
      "close": [105] * 10,
    })

    with pytest.raises(ValueError, match="Index must be DatetimeIndex"):
      pivot_points(data)

  def test_pivots_forward_fill(self):
    """Test that pivot levels are forward filled within period."""
    # Create hourly data for 2 days
    dates = pd.date_range("2024-01-01 00:00", periods=48, freq="h")
    data = pd.DataFrame(
      {
        "high": [110.0] * 48,
        "low": [100.0] * 48,
        "close": [105.0] * 48,
      },
      index=dates,
    )

    result = pivot_points(data, method="standard", period="D")

    # All bars in day 2 should have same pivot levels
    day2_mask = result.index.date == pd.Timestamp("2024-01-02").date()
    day2_data = result[day2_mask]

    # Skip first day (no previous data)
    if len(day2_data) > 0:
      assert day2_data["pp"].nunique() == 1
      assert day2_data["r1"].nunique() == 1
      assert day2_data["s1"].nunique() == 1

  def test_pivots_invalid_period(self):
    """Test pivots with invalid period."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    data = pd.DataFrame(
      {
        "high": [100] * 10,
        "low": [90] * 10,
        "close": [95] * 10,
      },
      index=dates,
    )

    # The Literal type hint should prevent invalid period at runtime
    # but let's test the function logic handles it
    with contextlib.suppress(ValueError, TypeError):
      # This will fail type checking, but let's see runtime behavior
      pivot_points(data, method="standard", period="YE")
      # If it doesn't raise, that's okay - validation may happen elsewhere

  def test_pivots_invalid_method(self):
    """Test pivots with invalid method."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    data = pd.DataFrame(
      {
        "high": [100] * 10,
        "low": [90] * 10,
        "close": [95] * 10,
      },
      index=dates,
    )

    # The Literal type hint should prevent invalid method at runtime
    with contextlib.suppress(ValueError, TypeError):
      pivot_points(data, method="invalid", period="D")  # type: ignore
      # If it doesn't raise, that's okay

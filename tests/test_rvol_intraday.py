"""Tests for intraday RVOL functionality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indikator.rvol import intraday_aggregate, rvol_intraday


class TestIntradayAggregate:
  """Tests for the generic intraday_aggregate function."""

  def test_basic_aggregation(self):
    """Test basic time-of-day aggregation."""
    # Create 3 days of data with 3 time slots per day
    # Use explicit dates to ensure same time slots repeat
    dates = pd.to_datetime([
      "2024-01-01 09:30",
      "2024-01-01 10:30",
      "2024-01-01 11:30",  # Day 1
      "2024-01-02 09:30",
      "2024-01-02 10:30",
      "2024-01-02 11:30",  # Day 2
      "2024-01-03 09:30",
      "2024-01-03 10:30",
      "2024-01-03 11:30",  # Day 3
    ])
    data = pd.DataFrame(
      {
        "volume": [
          100,
          200,
          300,  # Day 1: 9:30, 10:30, 11:30
          110,
          220,
          330,  # Day 2: same times
          120,
          240,
          360,  # Day 3: same times
        ]
      },
      index=dates,
    )

    # Calculate mean for each time slot
    result = intraday_aggregate(data["volume"], agg_func=pd.Series.mean, min_samples=2)

    # Day 1 should have NaN (no history)
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert pd.isna(result.iloc[2])

    # Day 2 should have NaN (only 1 sample, need min_samples=2)
    assert pd.isna(result.iloc[3])
    assert pd.isna(result.iloc[4])
    assert pd.isna(result.iloc[5])

    # Day 3 should have averages from days 1-2
    # 9:30 AM: (100 + 110) / 2 = 105
    # 10:30 AM: (200 + 220) / 2 = 210
    # 11:30 AM: (300 + 330) / 2 = 315
    assert result.iloc[6] == 105.0
    assert result.iloc[7] == 210.0
    assert result.iloc[8] == 315.0

  def test_different_agg_functions(self):
    """Test using different aggregation functions."""
    dates = pd.date_range("2024-01-01 09:30", periods=6, freq="1D")
    data = pd.DataFrame({"volume": [100, 200, 100, 200, 100, 200]}, index=dates)

    # All same time (9:30), different days
    mean_result = intraday_aggregate(
      data["volume"], agg_func=pd.Series.mean, min_samples=1
    )
    std_result = intraday_aggregate(
      data["volume"], agg_func=pd.Series.std, min_samples=2
    )

    # Mean should grow as we accumulate samples
    assert pd.isna(mean_result.iloc[0])  # No history
    assert mean_result.iloc[1] == 100  # Mean of [100]
    assert mean_result.iloc[2] == 150  # Mean of [100, 200]

    # Std requires at least 2 samples
    assert pd.isna(std_result.iloc[0])
    assert pd.isna(std_result.iloc[1])  # Only 1 sample
    np.testing.assert_almost_equal(
      std_result.iloc[2], np.std([100, 200], ddof=1)
    )  # Std of [100, 200]

  def test_lookback_days(self):
    """Test lookback_days parameter."""
    # Create 10 days of data, same time each day
    dates = pd.date_range("2024-01-01 09:30", periods=10, freq="1D")
    data = pd.DataFrame({"volume": list(range(100, 110))}, index=dates)

    # Use only last 3 days of history
    result = intraday_aggregate(
      data["volume"],
      agg_func=pd.Series.mean,
      lookback_days=3,
      min_samples=1,
    )

    # Last bar (day 10, value=109) should use days 7,8,9 (values 106,107,108)
    expected_mean = np.mean([106, 107, 108])
    assert result.iloc[-1] == expected_mean

  def test_non_datetime_index(self):
    """Test that non-DatetimeIndex raises error."""
    data = pd.DataFrame({"volume": [100, 200, 300]})  # Integer index

    with pytest.raises(ValueError, match="Index must be DatetimeIndex"):
      intraday_aggregate(data["volume"], agg_func=pd.Series.mean)

  def test_empty_dataframe(self):
    """Test empty DataFrame."""
    data = pd.DataFrame({"volume": []}, index=pd.DatetimeIndex([]))

    with pytest.raises(ValueError, match="Data must not be empty"):
      intraday_aggregate(data["volume"], agg_func=pd.Series.mean)


class TestRvolIntraday:
  """Tests for rvol_intraday function."""

  def test_basic_rvol_intraday(self):
    """Test basic intraday RVOL calculation."""
    # Create 3 days of data with consistent volume pattern
    dates = pd.to_datetime([
      "2024-01-01 09:30",
      "2024-01-01 10:30",
      "2024-01-01 11:30",  # Day 1
      "2024-01-02 09:30",
      "2024-01-02 10:30",
      "2024-01-02 11:30",  # Day 2
      "2024-01-03 09:30",
      "2024-01-03 10:30",
      "2024-01-03 11:30",  # Day 3
    ])
    data = pd.DataFrame(
      {
        "volume": [
          1000,
          2000,
          3000,  # Day 1: baseline
          1000,
          2000,
          3000,  # Day 2: same
          2000,
          4000,
          6000,  # Day 3: 2x spike
        ]
      },
      index=dates,
    )

    result = rvol_intraday(data["volume"], min_samples=2)

    # Day 3 should show 2x RVOL (current / avg of days 1-2)
    # 9:30: 2000 / ((1000+1000)/2) = 2.0
    # 10:30: 4000 / ((2000+2000)/2) = 2.0
    # 11:30: 6000 / ((3000+3000)/2) = 2.0
    assert result.iloc[6] == 2.0
    assert result.iloc[7] == 2.0
    assert result.iloc[8] == 2.0

  def test_market_open_vs_lunch(self):
    """Test that RVOL correctly handles different times of day."""
    # Simulate market pattern: high volume at open, low at lunch
    dates = pd.to_datetime([
      "2024-01-01 09:30",
      "2024-01-01 12:30",
      "2024-01-01 15:30",  # Day 1
      "2024-01-02 09:30",
      "2024-01-02 12:30",
      "2024-01-02 15:30",  # Day 2
    ])
    data = pd.DataFrame(
      {
        "volume": [
          5000,
          1000,
          5000,  # Day 1: open, lunch, close
          10000,
          1000,
          5000,  # Day 2: spike at open, normal lunch/close
        ]
      },
      index=dates,
    )

    result = rvol_intraday(data["volume"], min_samples=1)

    # Day 2 open: 10000 / 5000 = 2.0 (spike at open)
    # Day 2 lunch: 1000 / 1000 = 1.0 (normal for lunch)
    # Day 2 close: 5000 / 5000 = 1.0 (normal for close)
    assert result.iloc[3] == 2.0  # Open spike detected
    assert result.iloc[4] == 1.0  # Lunch normal
    assert result.iloc[5] == 1.0  # Close normal

  def test_insufficient_samples(self):
    """Test behavior with insufficient samples per time slot."""
    # Only 2 days, min_samples=3
    dates = pd.date_range("2024-01-01 09:30", periods=2, freq="1D")
    data = pd.DataFrame({"volume": [1000, 2000]}, index=dates)

    result = rvol_intraday(data["volume"], min_samples=3)

    # Should return 1.0 (neutral) when insufficient samples
    assert (result == 1.0).all()

  def test_non_datetime_index(self):
    """Test error with non-DatetimeIndex."""
    # Use string index which cannot be coerced to Datetime (easily)
    # Integers might be coerced to Timestamps (ns since epoch) if coerce=True
    data = pd.DataFrame({"volume": [100, 200, 300]}, index=["a", "b", "c"])

    with pytest.raises(ValueError):
      rvol_intraday(data["volume"])

  def test_empty_dataframe(self) -> None:
    """Empty input should raise ValueError."""
    empty_data = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
    with pytest.raises(ValueError, match="Data must not be empty"):
      rvol_intraday(empty_data)

  def test_division_by_zero_protection(self):
    """Test epsilon protection against zero division."""
    dates = pd.date_range("2024-01-01 09:30", periods=4, freq="1D")
    data = pd.DataFrame(
      {
        "volume": [
          0,
          0,
          100,
          200,
        ]  # First two days zero, then normal
      },
      index=dates,
    )

    result = rvol_intraday(data["volume"], min_samples=1, epsilon=1e-9)

    # Should handle zero average gracefully (return 1.0 default)
    assert result.iloc[2] == 1.0  # Avg is 0, use default

  def test_lookback_days(self):
    """Test lookback_days parameter."""
    # 10 days, but only use last 3 days
    dates = pd.date_range("2024-01-01 09:30", periods=10, freq="1D")
    # Volume increases each day
    data = pd.DataFrame({"volume": [100 * i for i in range(1, 11)]}, index=dates)

    result = rvol_intraday(data["volume"], lookback_days=3, min_samples=1)

    # Last bar (day 10, vol=1000) should compare to days 7-9 (700,800,900)
    # Average = 800, RVOL = 1000/800 = 1.25
    expected_rvol = 1000 / 800
    assert result.iloc[-1] == expected_rvol

  def test_returns_series(self):
    """Test that returns a Series."""
    dates = pd.date_range("2024-01-01 09:30", periods=5, freq="1h")
    data = pd.DataFrame(
      {
        "open": [100, 101, 102, 103, 104],
        "volume": [1000, 1100, 1200, 1300, 1400],
      },
      index=dates,
    )

    result = rvol_intraday(data["volume"])

    assert isinstance(result, pd.Series)
    assert result.name == "rvol_intraday"

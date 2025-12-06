"""Tests for intraday Z-Score functionality."""

import numpy as np
import pandas as pd
import pytest

from indikator.zscore import zscore_intraday


class TestZScoreIntraday:
    """Tests for zscore_intraday function."""

    def test_basic_zscore_intraday(self):
        """Test basic intraday Z-Score calculation."""
        # Create 3 days of data with consistent pattern
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
                "close": [
                    100,
                    110,
                    120,  # Day 1
                    100,
                    110,
                    120,  # Day 2
                    105,
                    120,
                    115,  # Day 3: +5, +10, -5 deviations
                ]
            },
            index=dates,
        )

        # We need std > 0, so let's ensure variance in history
        # Day 1 and 2 are identical, so std=0 if we only use them.
        # Intraday aggregate calculates mean and std for each time slot.
        # If values are identical (100, 100), std is 0.
        # Let's make Day 2 slightly different to have non-zero std.
        data.iloc[3] = 102  # 9:30 was 100, now 102. Mean=101, Std=1.414
        data.iloc[4] = 112  # 10:30 was 110, now 112. Mean=111, Std=1.414
        data.iloc[5] = 118  # 11:30 was 120, now 118. Mean=119, Std=1.414

        result = zscore_intraday(data["close"], min_samples=2)

        # Check Day 3 (indices 6, 7, 8)
        # 9:30 AM: History [100, 102]. Mean=101, Std=sqrt(2)=1.414. Value=105.
        # Z = (105 - 101) / 1.414 = 4 / 1.414 = 2.828
        np.testing.assert_almost_equal(result.iloc[6], 2.828, decimal=3)

    def test_market_open_vs_lunch(self):
        """Test that Z-Score correctly handles different volatilities."""
        # Simulate market pattern: high volatility at open, low at lunch
        dates = pd.to_datetime([
            "2024-01-01 09:30",
            "2024-01-01 12:30",  # Day 1
            "2024-01-02 09:30",
            "2024-01-02 12:30",  # Day 2
            "2024-01-03 09:30",
            "2024-01-03 12:30",  # Day 3
        ])
        data = pd.DataFrame(
            {
                "close": [
                    100,
                    100,  # Day 1
                    110,
                    101,  # Day 2: Open varies (100->110), Lunch stable (100->101)
                    120,
                    102,  # Day 3
                ]
            },
            index=dates,
        )

        # 9:30 History: [100, 110]. Mean=105, Std=7.07. Day 3 Value=120.
        # Z = (120 - 105) / 7.07 = 15 / 7.07 = 2.12
        # 12:30 History: [100, 101]. Mean=100.5, Std=0.707. Day 3 Value=102.
        # Z = (102 - 100.5) / 0.707 = 1.5 / 0.707 = 2.12

        result = zscore_intraday(data["close"], min_samples=2)

        np.testing.assert_almost_equal(result.iloc[4], 2.12, decimal=2)
        np.testing.assert_almost_equal(result.iloc[5], 2.12, decimal=2)

    def test_insufficient_samples(self):
        """Test behavior with insufficient samples."""
        dates = pd.date_range("2024-01-01 09:30", periods=2, freq="1D")
        data = pd.DataFrame({"close": [100, 105]}, index=dates)

        result = zscore_intraday(data["close"], min_samples=3)

        # Insufficient samples results in default neutral value (0.0)
        assert result.iloc[0] == 0.0
        assert result.iloc[1] == 0.0

    def test_non_datetime_index(self):
        """Test error with non-DatetimeIndex."""
        data = pd.DataFrame({"close": [100, 200, 300]})

        with pytest.raises(ValueError, match="Index must be DatetimeIndex"):
            zscore_intraday(data["close"])

    def test_empty_dataframe(self):
        """Test empty DataFrame handling."""
        data = pd.DataFrame({"close": []}, index=pd.DatetimeIndex([]))

        with pytest.raises(ValueError, match="Data must not be empty"):
            zscore_intraday(data["close"])

    def test_zero_std_protection(self):
        """Test protection against division by zero standard deviation."""
        dates = pd.date_range("2024-01-01 09:30", periods=3, freq="1D")
        data = pd.DataFrame(
            {
                "close": [100, 100, 100]  # No variance
            },
            index=dates,
        )

        result = zscore_intraday(data["close"], min_samples=2)

        # When std is 0 (no variance), z-score should default to 0.0
        # to avoid division by zero / infinity.
        assert result.iloc[2] == 0.0

    def test_returns_series(self):
        """Test that returns a Series."""
        dates = pd.date_range("2024-01-01 09:30", periods=5, freq="1h")
        data = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104],
                "close": [100, 101, 102, 103, 104],
            },
            index=dates,
        )

        result = zscore_intraday(data["close"])

        assert isinstance(result, pd.Series)
        assert result.name == "close_zscore_intraday"

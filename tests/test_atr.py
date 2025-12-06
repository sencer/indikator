"""Tests for ATR (Average True Range) indicator."""

import pandas as pd
import pytest

from indikator.atr import atr, atr_intraday


class TestATR:
    """Tests for ATR indicator."""

    def test_atr_basic(self):
        """Test ATR basic calculation."""
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
        })

        result = atr(data, window=5)

        # Check columns
        assert "atr" in result.columns
        assert "true_range" in result.columns

        # Check shape
        assert len(result) == len(data)

        # Check true range is calculated
        assert not result["true_range"].isna().all()

        # Check ATR is calculated after window
        assert result["atr"].isna().iloc[:4].all()  # First 4 bars should be NaN
        assert (
            not result["atr"].isna().iloc[4:].all()
        )  # After window should have values

        # Check ATR is positive
        assert (result["atr"].dropna() > 0).all()

    def test_atr_empty_data(self):
        """Test ATR with empty dataframe."""
        data = pd.DataFrame(columns=["high", "low", "close"]).astype(float)

        with pytest.raises(ValueError):
            atr(data)

    def test_atr_validation_missing_columns(self):
        """Test ATR validation with missing columns."""
        data = pd.DataFrame({"high": [100, 101], "low": [99, 100]})

        with pytest.raises(ValueError, match="Missing columns"):
            atr(data)

    def test_atr_manual_calculation(self):
        """Test ATR with manual calculation."""
        # Simple case where we can verify the true range
        data = pd.DataFrame({
            "high": [105.0, 108.0, 107.0],
            "low": [100.0, 103.0, 102.0],
            "close": [103.0, 106.0, 105.0],
        })

        result = atr(data, window=2)

        # First TR = high - low = 105 - 100 = 5
        assert result["true_range"].iloc[0] == 5.0

        # Second TR = max(108-103=5, |108-103|=5, |103-103|=0) = 5
        assert result["true_range"].iloc[1] == 5.0

        # Third TR = max(107-102=5, |107-106|=1, |102-106|=4) = 5
        assert result["true_range"].iloc[2] == 5.0

    def test_atr_window_parameter(self):
        """Test ATR with different window sizes."""
        data = pd.DataFrame({
            "high": [102.0, 104.0, 103.0, 106.0, 108.0] * 4,
            "low": [100.0, 101.0, 100.0, 103.0, 105.0] * 4,
            "close": [101.0, 103.0, 102.0, 105.0, 107.0] * 4,
        })

        result_short = atr(data, window=3)
        result_long = atr(data, window=10)

        # Short window should have values earlier
        assert result_short["atr"].notna().sum() > result_long["atr"].notna().sum()


class TestATRIntraday:
    """Tests for intraday ATR indicator."""

    def test_atr_intraday_basic(self):
        """Test intraday ATR basic calculation."""
        dates = pd.date_range("2024-01-01 09:30", periods=100, freq="5min")
        data = pd.DataFrame(
            {
                "high": [102.0] * 100,
                "low": [100.0] * 100,
                "close": [101.0] * 100,
            },
            index=dates,
        )

        result = atr_intraday(data, min_samples=3)

        # Check columns
        assert "atr_intraday" in result.columns
        assert "true_range" in result.columns

        # Check shape
        assert len(result) == len(data)

    def test_atr_intraday_empty_data(self):
        """Test intraday ATR with empty dataframe."""
        data = pd.DataFrame(columns=["high", "low", "close"]).astype(float)
        data.index = pd.DatetimeIndex([])

        with pytest.raises(ValueError):
            atr_intraday(data)

    def test_atr_intraday_requires_datetime_index(self):
        """Test intraday ATR requires DatetimeIndex."""
        data = pd.DataFrame({
            "high": [102.0, 104.0],
            "low": [100.0, 101.0],
            "close": [101.0, 103.0],
        })

        with pytest.raises(ValueError, match="Index must be DatetimeIndex"):
            atr_intraday(data)

    def test_atr_intraday_lookback_days(self):
        """Test intraday ATR with lookback period."""
        dates = pd.date_range("2024-01-01 09:30", periods=200, freq="5min")
        data = pd.DataFrame(
            {
                "high": [102.0] * 200,
                "low": [100.0] * 200,
                "close": [101.0] * 200,
            },
            index=dates,
        )

        result = atr_intraday(data, lookback_days=1, min_samples=3)

        # Check intraday ATR is calculated
        assert "atr_intraday" in result.columns
        assert len(result) == len(data)

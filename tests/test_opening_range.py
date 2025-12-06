"""Tests for Opening Range indicator."""

import pandas as pd
import pytest

from indikator.opening_range import opening_range


class TestOpeningRange:
    """Tests for Opening Range indicator."""

    def test_opening_range_basic(self):
        """Test opening range basic calculation."""
        # Create intraday data (5-minute bars)
        dates = pd.date_range("2024-01-02 09:30", periods=20, freq="5min")
        data = pd.DataFrame(
            {
                "high": [
                    102,
                    103,
                    104,
                    103,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                    116,
                    117,
                    118,
                    119,
                    120,
                ],
                "low": [
                    100,
                    101,
                    102,
                    101,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                    116,
                    117,
                    118,
                ],
                "close": [
                    101,
                    102,
                    103,
                    102,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                    116,
                    117,
                    118,
                    119,
                ],
            },
            index=dates,
        )

        result = opening_range(data, minutes=30)

        # Check columns exist
        assert "or_high" in result.columns
        assert "or_low" in result.columns
        assert "or_mid" in result.columns
        assert "or_range" in result.columns
        assert "or_breakout" in result.columns

        # Check shape
        assert len(result) == len(data)

        # Check OR is calculated (first 30 minutes = 6 bars at 5min intervals)
        # OR high should be max of first 6 bars
        expected_or_high = data["high"].iloc[:6].max()
        assert result["or_high"].iloc[0] == expected_or_high

        # OR low should be min of first 6 bars
        expected_or_low = data["low"].iloc[:6].min()
        assert result["or_low"].iloc[0] == expected_or_low

    def test_opening_range_empty_data(self):
        """Test opening range with empty dataframe."""
        data = pd.DataFrame(columns=["high", "low", "close"], dtype=float)
        data.index = pd.DatetimeIndex([])

        with pytest.raises(ValueError, match="Data must not be empty"):
            opening_range(data)

    def test_opening_range_mid_calculation(self):
        """Test opening range mid calculation."""
        dates = pd.date_range("2024-01-02 09:30", periods=10, freq="5min")
        data = pd.DataFrame(
            {
                "high": [110] * 10,
                "low": [100] * 10,
                "close": [105] * 10,
            },
            index=dates,
        )

        result = opening_range(data, minutes=30)

        # OR mid should be average of high and low
        expected_mid = (110 + 100) / 2
        assert result["or_mid"].iloc[0] == expected_mid

    def test_opening_range_range_calculation(self):
        """Test opening range range calculation."""
        dates = pd.date_range("2024-01-02 09:30", periods=10, freq="5min")
        data = pd.DataFrame(
            {
                "high": [110] * 10,
                "low": [100] * 10,
                "close": [105] * 10,
            },
            index=dates,
        )

        result = opening_range(data, minutes=30)

        # OR range should be high - low
        expected_range = 110 - 100
        assert result["or_range"].iloc[0] == expected_range

    def test_opening_range_breakout_above(self):
        """Test opening range breakout above."""
        dates = pd.date_range("2024-01-02 09:30", periods=10, freq="5min")
        data = pd.DataFrame(
            {
                "high": [102, 103, 104, 103, 105, 106, 117, 118, 119, 120],
                "low": [100, 101, 102, 101, 103, 104, 115, 116, 117, 118],
                "close": [101, 102, 103, 102, 104, 105, 116, 117, 118, 119],
            },
            index=dates,
        )

        result = opening_range(data, minutes=30)

        # First 6 bars (30 min = indices 0-5) define OR
        # Bar 7 (index 6) has close of 116, which should be > OR high
        or_high_val = result["or_high"].iloc[6]
        assert result["or_breakout"].iloc[6] == 1  # Above OR
        assert data["close"].iloc[6] > or_high_val

    def test_opening_range_breakout_below(self):
        """Test opening range breakout below."""
        dates = pd.date_range("2024-01-02 09:30", periods=10, freq="5min")
        data = pd.DataFrame(
            {
                "high": [102, 103, 104, 103, 105, 106, 95, 93, 92, 91],
                "low": [100, 101, 102, 101, 103, 104, 93, 91, 90, 89],
                "close": [101, 102, 103, 102, 104, 105, 94, 92, 91, 90],
            },
            index=dates,
        )

        result = opening_range(data, minutes=30)

        # First 6 bars (30 min = indices 0-5) define OR
        # Bar 7 (index 6) has close of 94, which should be < OR low
        or_low_val = result["or_low"].iloc[6]
        assert result["or_breakout"].iloc[6] == -1  # Below OR
        assert data["close"].iloc[6] < or_low_val

    def test_opening_range_inside_range(self):
        """Test opening range inside range."""
        dates = pd.date_range("2024-01-02 09:30", periods=10, freq="5min")
        data = pd.DataFrame(
            {
                "high": [102, 103, 104, 103, 105, 104, 103, 102, 101, 100],
                "low": [100, 101, 102, 101, 103, 102, 101, 100, 99, 98],
                "close": [101, 102, 103, 102, 104, 103, 102, 101, 100, 99],
            },
            index=dates,
        )

        result = opening_range(data, minutes=30)

        # Bar 6 (index 5) has close inside OR
        or_high_val = result["or_high"].iloc[5]
        or_low_val = result["or_low"].iloc[5]
        close_val = data["close"].iloc[5]

        assert result["or_breakout"].iloc[5] == 0  # Inside OR
        assert or_low_val <= close_val <= or_high_val

    def test_opening_range_multiple_sessions(self):
        """Test opening range across multiple sessions."""
        # Create 2 days of data
        dates1 = pd.date_range("2024-01-02 09:30", periods=10, freq="5min")
        dates2 = pd.date_range("2024-01-03 09:30", periods=10, freq="5min")
        dates = dates1.append(dates2)

        data = pd.DataFrame(
            {
                "high": [110] * 10 + [120] * 10,
                "low": [100] * 10 + [110] * 10,
                "close": [105] * 10 + [115] * 10,
            },
            index=dates,
        )

        result = opening_range(data, minutes=30)

        # Day 1 OR high should be 110
        assert result["or_high"].iloc[0] == 110
        # Day 2 OR high should be 120
        assert result["or_high"].iloc[10] == 120

    def test_opening_range_custom_period(self):
        """Test opening range with custom period."""
        dates = pd.date_range("2024-01-02 09:30", periods=20, freq="5min")
        data = pd.DataFrame(
            {
                "high": list(range(100, 120)),
                "low": list(range(90, 110)),
                "close": list(range(95, 115)),
            },
            index=dates,
        )

        # 15 minute OR (3 bars)
        result = opening_range(data, minutes=15)

        # OR should be based on first 3 bars
        expected_or_high = data["high"].iloc[:3].max()
        expected_or_low = data["low"].iloc[:3].min()

        assert result["or_high"].iloc[0] == expected_or_high
        assert result["or_low"].iloc[0] == expected_or_low

    def test_opening_range_custom_session_start(self):
        """Test opening range with custom session start."""
        # Start at 10:00 instead of 09:30
        dates = pd.date_range("2024-01-02 10:00", periods=10, freq="5min")
        data = pd.DataFrame(
            {
                "high": [110] * 10,
                "low": [100] * 10,
                "close": [105] * 10,
            },
            index=dates,
        )

        result = opening_range(data, minutes=30, session_start="10:00")

        # Should calculate OR starting from 10:00
        assert not result["or_high"].isna().all()

    def test_opening_range_validation_missing_columns(self):
        """Test opening range validation with missing columns."""
        dates = pd.date_range("2024-01-02 09:30", periods=10, freq="5min")
        data = pd.DataFrame(
            {
                "high": [110] * 10,
                "low": [100] * 10,
            },
            index=dates,
        )

        with pytest.raises(ValueError, match="Missing columns"):
            opening_range(data)

    def test_opening_range_validation_not_datetime_index(self):
        """Test opening range validation with non-datetime index."""
        data = pd.DataFrame({
            "high": [110] * 10,
            "low": [100] * 10,
            "close": [105] * 10,
        })

        with pytest.raises(ValueError, match="Index must be DatetimeIndex"):
            opening_range(data)

    def test_opening_range_forward_fill(self):
        """Test that OR levels are forward filled throughout the session."""
        dates = pd.date_range("2024-01-02 09:30", periods=10, freq="5min")
        data = pd.DataFrame(
            {
                "high": [110] * 10,
                "low": [100] * 10,
                "close": [105] * 10,
            },
            index=dates,
        )

        result = opening_range(data, minutes=30)

        # All bars in the session should have the same OR levels
        assert result["or_high"].nunique() == 1
        assert result["or_low"].nunique() == 1
        assert result["or_mid"].nunique() == 1
        assert result["or_range"].nunique() == 1

"""Tests for MFI (Money Flow Index) indicator."""

import pandas as pd
import pytest

from indikator.mfi import mfi


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

        # Check columns
        assert "mfi" in result.columns
        assert "typical_price" in result.columns

        # Check shape
        assert len(result) == len(data)

        # Check MFI is calculated after window
        assert result["mfi"].isna().iloc[:5].all()
        assert not result["mfi"].isna().iloc[5:].all()

        # Check MFI is in range [0, 100]
        assert (result["mfi"].dropna() >= 0).all()
        assert (result["mfi"].dropna() <= 100).all()

    def test_mfi_typical_price(self):
        """Test MFI typical price calculation."""
        data = pd.DataFrame({
            "high": [102.0, 104.0],
            "low": [100.0, 102.0],
            "close": [101.0, 103.0],
            "volume": [1000.0, 1200.0],
        })

        result = mfi(data, window=2)

        # Typical price = (H + L + C) / 3
        expected_typical = (data["high"] + data["low"] + data["close"]) / 3
        pd.testing.assert_series_equal(
            result["typical_price"], expected_typical, check_names=False
        )

    def test_mfi_empty_data(self):
        """Test MFI with empty dataframe."""
        data = pd.DataFrame(columns=["high", "low", "close", "volume"]).astype(float)

        with pytest.raises(ValueError, match="Data must not be empty"):
            mfi(data)

    def test_mfi_validation_missing_columns(self):
        """Test MFI validation with missing columns."""
        data = pd.DataFrame({
            "high": [102.0, 104.0],
            "low": [100.0, 102.0],
            "close": [101.0, 103.0],
        })

        with pytest.raises(ValueError, match="Missing columns"):
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
        assert result_short["mfi"].notna().sum() > result_long["mfi"].notna().sum()

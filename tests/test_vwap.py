"""Tests for VWAP indicators."""

import pandas as pd
import pytest

from indikator.vwap import vwap, vwap_anchored


class TestVWAP:
  """Tests for VWAP indicator."""

  def test_vwap_basic(self):
    """Test VWAP basic calculation."""
    dates = pd.date_range("2024-01-01 09:30", periods=10, freq="5min")
    data = pd.DataFrame(
      {
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
        ],
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
        ],
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
        ],
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
        ],
      },
      index=dates,
    )

    result = vwap(data)

    # Check columns
    assert "vwap" in result.columns
    assert "typical_price" in result.columns

    # Check shape
    assert len(result) == len(data)

    # Check VWAP is calculated
    assert not result["vwap"].isna().any()

    # Check typical price is calculated correctly
    expected_typical = (data["high"] + data["low"] + data["close"]) / 3
    pd.testing.assert_series_equal(
      result["typical_price"], expected_typical, check_names=False
    )

  def test_vwap_manual_calculation(self):
    """Test VWAP with manual calculation."""
    dates = pd.date_range("2024-01-01 09:30", periods=3, freq="5min")
    data = pd.DataFrame(
      {
        "high": [102.0, 104.0, 106.0],
        "low": [100.0, 102.0, 104.0],
        "close": [101.0, 103.0, 105.0],
        "volume": [100.0, 100.0, 100.0],
      },
      index=dates,
    )

    result = vwap(data, session_freq="D")

    # Typical prices: (102+100+101)/3=101, (104+102+103)/3=103, (106+104+105)/3=105
    # VWAP at bar 0: 101 * 100 / 100 = 101
    assert abs(result["vwap"].iloc[0] - 101.0) < 0.01

    # VWAP at bar 1: (101*100 + 103*100) / 200 = 102
    assert abs(result["vwap"].iloc[1] - 102.0) < 0.01

    # VWAP at bar 2: (101*100 + 103*100 + 105*100) / 300 = 103
    assert abs(result["vwap"].iloc[2] - 103.0) < 0.01

  def test_vwap_session_reset(self):
    """Test VWAP resets at session boundaries."""
    dates = pd.date_range("2024-01-01 09:30", periods=20, freq="1h")
    data = pd.DataFrame(
      {
        "high": [102.0] * 20,
        "low": [100.0] * 20,
        "close": [101.0] * 20,
        "volume": [1000.0] * 20,
      },
      index=dates,
    )

    result = vwap(data, session_freq="D")

    # VWAP should reset each day
    # Since prices are constant, VWAP should equal typical price
    expected_typical = (data["high"] + data["low"] + data["close"]) / 3
    pd.testing.assert_series_equal(result["vwap"], expected_typical, check_names=False)

  def test_vwap_empty_data(self):
    """Test VWAP with empty dataframe."""
    data = pd.DataFrame(columns=["high", "low", "close", "volume"]).astype(float)
    data.index = pd.DatetimeIndex([])

    with pytest.raises(ValueError, match="Data must not be empty"):
      vwap(data)

  def test_vwap_requires_datetime_index(self):
    """Test VWAP requires DatetimeIndex."""
    data = pd.DataFrame({
      "high": [102.0, 104.0],
      "low": [100.0, 101.0],
      "close": [101.0, 103.0],
      "volume": [1000.0, 1200.0],
    })

    with pytest.raises(ValueError, match="Index must be DatetimeIndex"):
      vwap(data)


class TestVWAPAnchored:
  """Tests for anchored VWAP indicator."""

  def test_vwap_anchored_by_index(self):
    """Test anchored VWAP with index anchor."""
    data = pd.DataFrame({
      "high": [102.0, 104.0, 106.0, 108.0, 110.0],
      "low": [100.0, 102.0, 104.0, 106.0, 108.0],
      "close": [101.0, 103.0, 105.0, 107.0, 109.0],
      "volume": [100.0, 100.0, 100.0, 100.0, 100.0],
    })

    result = vwap_anchored(data, anchor_index=2)

    # Check columns
    assert "vwap_anchored" in result.columns
    assert "typical_price" in result.columns

    # Before anchor should be NaN
    assert result["vwap_anchored"].iloc[:2].isna().all()

    # After anchor should have values
    assert not result["vwap_anchored"].iloc[2:].isna().any()

  def test_vwap_anchored_by_datetime(self):
    """Test anchored VWAP with datetime anchor."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    data = pd.DataFrame(
      {
        "high": [102.0, 104.0, 106.0, 108.0, 110.0],
        "low": [100.0, 102.0, 104.0, 106.0, 108.0],
        "close": [101.0, 103.0, 105.0, 107.0, 109.0],
        "volume": [100.0, 100.0, 100.0, 100.0, 100.0],
      },
      index=dates,
    )

    result = vwap_anchored(data, anchor_datetime="2024-01-03")

    # Before anchor should be NaN
    assert result["vwap_anchored"].iloc[:2].isna().all()

    # After anchor should have values
    assert not result["vwap_anchored"].iloc[2:].isna().any()

  def test_vwap_anchored_validation(self):
    """Test anchored VWAP parameter validation."""
    data = pd.DataFrame({
      "high": [102.0, 104.0],
      "low": [100.0, 102.0],
      "close": [101.0, 103.0],
      "volume": [100.0, 100.0],
    })

    # Should raise error if neither anchor provided
    with pytest.raises(ValueError, match="Must provide either"):
      vwap_anchored(data)

    # Should raise error if both anchors provided
    with pytest.raises(ValueError, match="Cannot provide both"):
      vwap_anchored(data, anchor_index=0, anchor_datetime="2024-01-01")

    # Should raise error if anchor out of range
    with pytest.raises(ValueError, match="out of range"):
      vwap_anchored(data, anchor_index=10)

  def test_vwap_anchored_empty_data(self):
    """Test anchored VWAP with empty dataframe."""
    data = pd.DataFrame(columns=["high", "low", "close", "volume"]).astype(float)

    with pytest.raises(ValueError, match="Data must not be empty"):
      vwap_anchored(data, anchor_index=0)

  def test_vwap_weekly_session(self):
    """Test VWAP with weekly session frequency."""
    dates = pd.date_range("2024-01-01", periods=14, freq="D")
    data = pd.DataFrame(
      {
        "high": [110.0] * 14,
        "low": [100.0] * 14,
        "close": [105.0] * 14,
        "volume": [1000.0] * 14,
      },
      index=dates,
    )

    result = vwap(data, session_freq="W")

    assert "vwap" in result.columns
    assert len(result) == len(data)

  def test_vwap_monthly_session(self):
    """Test VWAP with monthly session frequency."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    data = pd.DataFrame(
      {
        "high": [110.0] * 60,
        "low": [100.0] * 60,
        "close": [105.0] * 60,
        "volume": [1000.0] * 60,
      },
      index=dates,
    )

    result = vwap(data, session_freq="ME")

    assert "vwap" in result.columns
    assert len(result) == len(data)

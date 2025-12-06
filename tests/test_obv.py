"""Tests for OBV (On-Balance Volume) indicator."""

import pandas as pd
import pytest

from indikator.obv import obv


class TestOBV:
  """Tests for OBV indicator."""

  def test_obv_basic(self):
    """Test OBV basic calculation."""
    data = pd.DataFrame({
      "close": [100.0, 102.0, 101.0, 103.0, 105.0],
      "volume": [1000.0, 1200.0, 900.0, 1500.0, 1100.0],
    })

    result = obv(data)

    # Check column
    assert "obv" in result.columns

    # Check shape
    assert len(result) == len(data)

    # Check OBV is calculated
    assert not result["obv"].isna().any()

  def test_obv_manual_calculation(self):
    """Test OBV with manual calculation."""
    data = pd.DataFrame({
      "close": [100.0, 102.0, 101.0, 103.0, 105.0],
      "volume": [1000.0, 1200.0, 900.0, 1500.0, 1100.0],
    })

    result = obv(data)

    # First bar: OBV = volume = 1000
    assert result["obv"].iloc[0] == 1000.0

    # Second bar: close up (102 > 100), OBV = 1000 + 1200 = 2200
    assert result["obv"].iloc[1] == 2200.0

    # Third bar: close down (101 < 102), OBV = 2200 - 900 = 1300
    assert result["obv"].iloc[2] == 1300.0

    # Fourth bar: close up (103 > 101), OBV = 1300 + 1500 = 2800
    assert result["obv"].iloc[3] == 2800.0

    # Fifth bar: close up (105 > 103), OBV = 2800 + 1100 = 3900
    assert result["obv"].iloc[4] == 3900.0

  def test_obv_flat_price(self):
    """Test OBV when price is flat."""
    data = pd.DataFrame({
      "close": [100.0, 100.0, 100.0],
      "volume": [1000.0, 1200.0, 900.0],
    })

    result = obv(data)

    # OBV should not change when price is flat
    assert result["obv"].iloc[0] == 1000.0  # First bar
    assert result["obv"].iloc[1] == 1000.0  # No change (price flat)
    assert result["obv"].iloc[2] == 1000.0  # No change (price flat)

  def test_obv_empty_data(self):
    """Test OBV with empty dataframe."""
    data = pd.DataFrame(columns=["close", "volume"]).astype(float)

    with pytest.raises(ValueError, match="Data must not be empty"):
      obv(data)

  def test_obv_validation_missing_columns(self):
    """Test OBV validation with missing columns."""
    data = pd.DataFrame({"close": [100.0, 102.0]})

    with pytest.raises(ValueError, match="Missing columns"):
      obv(data)

  def test_obv_cumulative(self):
    """Test OBV is cumulative."""
    data = pd.DataFrame({
      "close": [100.0, 102.0, 104.0, 106.0, 108.0],
      "volume": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
    })

    result = obv(data)

    # All prices increasing, OBV should be increasing
    assert (result["obv"].diff().dropna() > 0).all()

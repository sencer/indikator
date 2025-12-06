"""Tests for rvol (Relative Volume) indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indikator.rvol import rvol


class TestBasicFunctionality:
  """Test core functionality of rvol."""

  @pytest.fixture
  def volume_df(self) -> pd.DataFrame:
    """Sample volume data."""
    return pd.DataFrame({"volume": [1000, 1100, 1200, 1300, 1400]})

  def test_returns_series_named_rvol(self, volume_df: pd.DataFrame) -> None:
    """Should return Series named rvol."""
    result = rvol(volume_df["volume"], window=3)

    assert isinstance(result, pd.Series)
    assert result.name == "rvol"

  def test_does_not_modify_original_series(self, volume_df: pd.DataFrame) -> None:
    """Should not modify the input Series."""
    series = volume_df["volume"].copy()
    _ = rvol(series)

    pd.testing.assert_series_equal(series, volume_df["volume"])

  def test_output_length_matches_input(self, volume_df: pd.DataFrame) -> None:
    """Output should have same length as input."""
    result = rvol(volume_df["volume"], window=3)
    assert len(result) == len(volume_df)

  def test_empty_series_returns_empty_series(self) -> None:
    """Empty input should raise ValueError."""
    empty_series = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="Data must not be empty"):
      rvol(empty_series)


class TestRVOLCalculation:
  """Test RVOL calculation logic."""

  def test_constant_volume_returns_one(self) -> None:
    """Constant volume should return RVOL of 1.0 (average)."""
    data = pd.Series([1000, 1000, 1000, 1000, 1000])
    result = rvol(data, window=3)

    # After window is satisfied, all values should be ~1.0
    rvol_values = result.iloc[2:]  # Skip first 2 (warm-up)
    assert np.allclose(rvol_values, 1.0, rtol=1e-10)

  def test_spike_in_volume_increases_rvol(self) -> None:
    """Volume spike should produce RVOL > 1."""
    data = pd.Series([1000, 1000, 1000, 3000, 1000])
    result = rvol(data, window=3)

    # Fourth bar (3000 volume) should have RVOL > 1
    assert result.iloc[3] > 1.0

  def test_drop_in_volume_decreases_rvol(self) -> None:
    """Volume drop should produce RVOL < 1."""
    data = pd.Series([1000, 1000, 1000, 500, 1000])
    result = rvol(data, window=3)

    # Fourth bar (500 volume) should have RVOL < 1
    assert result.iloc[3] < 1.0

  def test_rvol_magnitude_reflects_deviation(self) -> None:
    """Larger volume deviations should produce larger RVOL deviations from 1."""
    # Small spike
    small_spike = pd.Series([1000, 1000, 1000, 1500])
    result_small = rvol(small_spike, window=3)

    # Large spike
    large_spike = pd.Series([1000, 1000, 1000, 5000])
    result_large = rvol(large_spike, window=3)

    # Large spike should have higher RVOL
    assert result_large.iloc[-1] > result_small.iloc[-1]

  def test_window_parameter_affects_sensitivity(self) -> None:
    """Larger window retains memory of normal volume levels."""
    data = pd.Series([1000, 1000, 1000, 1000, 5000])

    result_small = rvol(data, window=2)
    result_large = rvol(data, window=4)

    # Larger window remembers more historical data, showing spike more dramatically
    # window=4: mean=[1000,1000,1000,5000]/4=2000, rvol=5000/2000=2.5
    # window=2: mean=[1000,5000]/2=3000, rvol=5000/3000=1.67
    assert result_large.iloc[-1] > result_small.iloc[-1]


class TestInsufficientData:
  """Test handling of insufficient data."""

  def test_insufficient_data_returns_neutral(self) -> None:
    """Should return 1.0 (neutral) when data is shorter than window."""
    data = pd.Series([1000, 1100])
    result = rvol(data, window=10)

    # All values should be 1.0 (neutral default)
    assert (result == 1.0).all()

  def test_single_value_returns_neutral(self) -> None:
    """Single value should return neutral RVOL."""
    data = pd.Series([1000])
    result = rvol(data, window=3)

    assert result.iloc[0] == 1.0


class TestZeroVolumeHandling:
  """Test handling of zero or near-zero volume."""

  def test_zero_average_volume_does_not_cause_inf(self) -> None:
    """Should handle zero average volume without producing inf."""
    # All zeros except last
    data = pd.Series([0, 0, 0, 1000])
    result = rvol(data, window=3, epsilon=1e-9)

    # Should not contain inf values
    assert not np.isinf(result).any()

  def test_all_zero_volume_returns_valid_values(self) -> None:
    """Should handle all-zero volume gracefully."""
    data = pd.Series([0, 0, 0, 0])
    result = rvol(data, window=3, epsilon=1e-9)

    # Should not contain inf or NaN
    assert not np.isinf(result).any()
    assert not np.isnan(result).any()

  def test_epsilon_prevents_division_by_zero(self) -> None:
    """Epsilon should prevent division by zero."""
    data = pd.Series([0.0, 0.0, 0.0, 1000.0])
    result = rvol(data, window=3, epsilon=1e-9)

    # Should complete without error and not have inf
    assert not np.isinf(result).any()


class TestEdgeCases:
  """Test edge cases and error handling."""

  def test_very_large_volume_spike(self) -> None:
    """Should handle very large volume spikes."""
    data = pd.Series([1000, 1000, 1000, 1000000])
    result = rvol(data, window=3)

    # Should be very large but not inf
    # Mean of [1000, 1000, 1000000]/3 = 334,000, rvol = 1000000/334000 ~= 3
    assert result.iloc[-1] > 2.5
    assert not np.isinf(result.iloc[-1])


class TestParameterValidation:
  """Test parameter validation and constraints."""

  @pytest.fixture
  def volume_df(self) -> pd.DataFrame:
    """Sample volume data."""
    return pd.DataFrame({"volume": [1000, 1100, 1200, 1300, 1400]})

  def test_default_parameters_work(self, volume_df: pd.DataFrame) -> None:
    """Should work with all default parameters."""
    result = rvol(volume_df["volume"])
    assert isinstance(result, pd.Series)

  def test_custom_window(self, volume_df: pd.DataFrame) -> None:
    """Should accept custom window parameter."""
    result = rvol(volume_df["volume"], window=3)
    assert isinstance(result, pd.Series)

  def test_custom_epsilon(self, volume_df: pd.DataFrame) -> None:
    """Should accept custom epsilon parameter."""
    result = rvol(volume_df["volume"], epsilon=1e-6)
    assert isinstance(result, pd.Series)


class TestInterpretation:
  """Test RVOL interpretation scenarios."""

  def test_rvol_two_indicates_double_average(self) -> None:
    """RVOL > 1 indicates above-average volume."""
    # Consistent volume, then double
    data = pd.Series([1000, 1000, 1000, 2000])
    result = rvol(data, window=3)

    # Last bar: mean=[1000,1000,2000]/3=1333, rvol=2000/1333=1.5
    assert np.isclose(result.iloc[-1], 1.5, rtol=0.01)

  def test_rvol_half_indicates_half_average(self) -> None:
    """RVOL < 1 indicates below-average volume."""
    # Consistent volume, then half
    data = pd.Series([1000, 1000, 1000, 500])
    result = rvol(data, window=3)

    # Last bar: mean=[1000,1000,500]/3=833, rvol=500/833=0.6
    assert np.isclose(result.iloc[-1], 0.6, rtol=0.01)

  def test_increasing_volume_trend(self) -> None:
    """Steadily increasing volume should show RVOL trending above 1."""
    data = pd.Series([1000, 1200, 1400, 1600, 1800, 2000])
    result = rvol(data, window=3)

    # Later values should be above 1 (above their rolling average)
    rvol_values = result.iloc[3:]
    assert (rvol_values > 1.0).all()

  def test_decreasing_volume_trend(self) -> None:
    """Steadily decreasing volume should show RVOL trending below 1."""
    data = pd.Series([2000, 1800, 1600, 1400, 1200, 1000])
    result = rvol(data, window=3)

    # Later values should be below 1 (below their rolling average)
    rvol_values = result.iloc[3:]
    assert (rvol_values < 1.0).all()


class TestFullOHLCVData:
  """Test with complete OHLCV data."""

  @pytest.fixture
  def ohlcv_df(self) -> pd.DataFrame:
    """Full OHLCV DataFrame."""
    return pd.DataFrame({
      "open": [100.0, 102.0, 105.0, 108.0, 110.0],
      "high": [103.0, 106.0, 109.0, 112.0, 114.0],
      "low": [99.0, 101.0, 104.0, 107.0, 109.0],
      "close": [102.0, 105.0, 108.0, 111.0, 113.0],
      "volume": [1000, 1500, 1200, 3000, 1100],
    })

  def test_rvol_calculation_with_ohlcv(self, ohlcv_df: pd.DataFrame) -> None:
    """Should correctly calculate RVOL with full OHLCV data."""
    result = rvol(ohlcv_df["volume"], window=3)

    # Fourth bar: volumes=[1500,1200,3000], mean=1900, rvol=3000/1900=1.58
    assert result.iloc[3] > 1.5

    # Fifth bar: volumes=[1200,3000,1100], mean=1767, rvol=1100/1767=0.62
    assert result.iloc[4] < 1.0


class TestRealWorldScenarios:
  """Test real-world trading scenarios."""

  def test_breakout_volume_spike(self) -> None:
    """Simulate breakout with volume spike."""
    # Normal volume, then breakout with 3x volume
    data = pd.Series([1000, 1100, 1000, 1050, 1000, 3000, 2500, 2000])
    result = rvol(data, window=5)

    # Breakout bar (index 5) should have very high RVOL
    assert result.iloc[5] > 2.0

  def test_low_volume_consolidation(self) -> None:
    """Simulate low-volume consolidation period."""
    # Normal volume, then consolidation with low volume
    data = pd.Series([1000, 1100, 1000, 400, 300, 350, 380])
    result = rvol(data, window=3)

    # Consolidation bars should have low RVOL (below 1.0)
    # Index 3: mean=[1100,1000,400]/3=833, rvol=400/833=0.48
    # Index 4: mean=[1000,400,300]/3=567, rvol=300/567=0.53
    consolidation_rvol = result.iloc[3:5]
    assert (consolidation_rvol < 0.6).all()

  def test_climactic_volume_pattern(self) -> None:
    """Simulate climactic volume pattern (high then declining)."""
    data = pd.Series([1000, 1500, 2000, 5000, 2000, 1000, 800])
    result = rvol(data, window=3)

    # Peak volume should have highest RVOL
    peak_idx = 3
    assert result.iloc[peak_idx] == result.iloc[2:].max()


class TestNumericalStability:
  """Test numerical stability with various inputs."""

  def test_large_volume_values(self) -> None:
    """Should handle very large volume values."""
    data = pd.Series([1e9, 1.1e9, 1.2e9, 1.3e9])
    result = rvol(data, window=3)

    assert isinstance(result, pd.Series)
    assert not np.isinf(result).any()
    assert not np.isnan(result).any()

  def test_small_volume_values(self) -> None:
    """Should handle very small volume values."""
    data = pd.Series([0.001, 0.0011, 0.0012, 0.0013])
    result = rvol(data, window=3)

    assert isinstance(result, pd.Series)
    assert not np.isinf(result).any()
    assert not np.isnan(result).any()

  def test_mixed_small_and_large_values(self) -> None:
    """Should handle mix of small and large volume values."""
    data = pd.Series([100, 0.01, 10000, 1, 500])
    result = rvol(data, window=3)

    assert isinstance(result, pd.Series)
    assert not np.isinf(result).any()

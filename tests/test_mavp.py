import numpy as np
import pandas as pd
import pytest
import talib
from indikator.mavp import mavp


def test_mavp_matches_talib_sma():
  np.random.seed(42)
  periods_count = 100
  data = pd.Series(np.random.randn(periods_count), name="data").astype(np.float64)
  # Generate random periods between 2 and 30
  periods = pd.Series(
    np.random.randint(2, 30, size=periods_count), name="periods"
  ).astype(np.float64)

  # matype=0 is SMA
  result = mavp(data, periods, minperiod=2, maxperiod=30, matype=0)
  expected = talib.MAVP(
    data.values, periods.values, minperiod=2, maxperiod=30, matype=0
  )

  s_expected = pd.Series(expected, index=data.index, name="mavp")

  # TA-Lib behavior is unstable/NaN during the first `maxperiod` (or close to it),
  # whereas our implementation is precise. We compare only the stable region.
  # Safe bet: slice valid data from maxperiod onwards.

  pd.testing.assert_series_equal(result.iloc[30:], s_expected.iloc[30:], atol=1e-10)


def test_mavp_clamping_logic():
  # Verify minperiod / maxperiod clamping works as expected.
  # Create periods requiring clamping.
  data = pd.Series(np.ones(10), dtype=float)
  periods = pd.Series([1, 1, 1, 100, 100, 5, 5, 5, 5, 5], dtype=float)

  # Min period 2, Max period 5
  # indices 0-2 (periods=1) should be clamped to 2
  # indices 3-4 (periods=100) should be clamped to 5

  result = mavp(data, periods, minperiod=2, maxperiod=5, matype=0)

  # Expected behavior:
  # At index 2: period clamped to 2. Window [1, 2]. Average of ones is 1.
  # At index 4: period clamped to 5. Window [0, 4]. Average of ones is 1.
  # Result should be all ones (except initial NaN).

  # Check NaN handling
  # For period=2, index 0 is NaN (insufficient). Index 1 is valid (0,1).
  assert np.isnan(result.iloc[0])
  assert result.iloc[1] == 1.0
  assert result.iloc[4] == 1.0


def test_mavp_variable_periods():
  # Manual verification of calculation
  data = pd.Series([10, 20, 30, 40, 50], dtype=float)
  periods = pd.Series([2, 2, 3, 2, 5], dtype=float)

  result = mavp(data, periods, minperiod=2, maxperiod=5)

  # Index 0: period 2. NaN.
  assert np.isnan(result.iloc[0])

  # Index 1: period 2. Window [10, 20]. Avg 15.
  assert result.iloc[1] == 15.0

  # Index 2: period 3. Window [10, 20, 30]. Avg 20.
  assert result.iloc[2] == 20.0

  # Index 3: period 2. Window [30, 40]. Avg 35.
  assert result.iloc[3] == 35.0

  # Index 4: period 5. Window [10, 20, 30, 40, 50]. Avg 30.
  assert result.iloc[4] == 30.0


def test_mavp_nans_in_periods():
  data = pd.Series([10, 20, 30], dtype=float)
  periods = pd.Series([2, np.nan, 2], dtype=float)

  result = mavp(data, periods, minperiod=2, maxperiod=5)

  assert np.isnan(result.iloc[1])
  assert result.iloc[2] == 25.0  # (20+30)/2

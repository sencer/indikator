"""Average True Range (ATR) indicator module.

This module provides ATR calculation, a volatility indicator that measures
the average range of price movement. Essential for position sizing and
stop-loss placement.
"""

from typing import Literal

from hipr import Ge, Hyper, configurable
import numpy as np
import pandas as pd
from pdval import (
  Datetime,
  HasColumns,
  Index,
  Validated,
  validated,
)

from indikator._atr_numba import compute_atr_numba, compute_true_range_numba
from indikator._intraday import intraday_aggregate

# Default minimum samples per time slot for intraday ATR
_DEFAULT_MIN_SAMPLES = 3


@configurable
@validated
def atr(
  data: Validated[pd.DataFrame, HasColumns[Literal["high", "low", "close"]]],
  window: Hyper[int, Ge[1]] = 14,
) -> pd.DataFrame:
  """Calculate Average True Range (ATR).

  ATR measures market volatility by calculating the average of true ranges
  over a specified period. Uses Wilder's smoothing method for a smoother output.

  The True Range is the greatest of:
  - Current high - current low
  - |Current high - previous close|
  - |Current low - previous close|

  ATR is essential for:
  - Position sizing (risk-adjusted position sizes)
  - Stop-loss placement (volatility-based stops)
  - Identifying breakout potential (volatility expansion)
  - Trend strength assessment (higher ATR = stronger trend)

  Uses Wilder's smoothing (similar to EMA):
  ATR[i] = (ATR[i-1] * (window-1) + TR[i]) / window

  Features:
  - Numba-optimized for performance
  - Handles edge cases (first bar, insufficient data)
  - Returns both ATR and True Range values
  - Standard 14-period default (Wilder's original)

  Args:
    data: OHLCV DataFrame with 'high', 'low', 'close' columns
    window: Smoothing period (default: 14, Wilder's original)

  Returns:
    DataFrame with 'atr' and 'true_range' columns added

  Raises:
    ValueError: If required columns are missing or data contains NaN/Inf

  Example:
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'high': [102, 104, 103, 106, 108],
    ...     'low': [100, 101, 100, 103, 105],
    ...     'close': [101, 103, 102, 105, 107]
    ... })
    >>> result = atr(data, window=3)
    >>> # Returns DataFrame with 'atr' and 'true_range' columns
  """
  # Convert to numpy arrays for Numba
  highs = np.asarray(data["high"].values, dtype=np.float64)
  lows = np.asarray(data["low"].values, dtype=np.float64)
  closes = np.asarray(data["close"].values, dtype=np.float64)

  # Calculate true range and ATR using Numba-optimized functions
  true_ranges = compute_true_range_numba(highs, lows, closes)
  atr_values = compute_atr_numba(true_ranges, window)

  # Create result dataframe
  data_copy = data.copy()
  data_copy["true_range"] = true_ranges
  data_copy["atr"] = atr_values

  return data_copy


@configurable
@validated
def atr_intraday(
  data: Validated[
    pd.DataFrame,
    HasColumns[Literal["high", "low", "close"]],
    Index[Datetime],
  ],
  lookback_days: int | None = None,
  min_samples: Hyper[int, Ge[2]] = _DEFAULT_MIN_SAMPLES,
) -> pd.DataFrame:
  """Calculate time-of-day adjusted ATR (intraday volatility).

  Compares current volatility to the historical average volatility for that
  specific time of day. This accounts for intraday volatility patterns:
  - Market open (9:30-10:00) typically has high volatility
  - Lunch (12:00-13:00) typically has low volatility
  - Market close (15:30-16:00) typically has high volatility

  Regular ATR might show "high volatility" during market open even when it's
  normal for that time. Intraday ATR correctly identifies "high for this time
  of day".

  Features:
  - Accounts for natural intraday volatility patterns
  - Configurable lookback period (None = use all history)
  - Requires minimum samples per time slot for reliability
  - Returns both intraday ATR and True Range

  Args:
    data: OHLCV DataFrame with DatetimeIndex and 'high', 'low', 'close' columns
    lookback_days: Number of days to look back (None = use all history)
    min_samples: Minimum historical samples required per time slot

  Returns:
    DataFrame with 'atr_intraday' and 'true_range' columns added

  Raises:
    ValueError: If required columns missing or index is not DatetimeIndex

  Example:
    >>> import pandas as pd
    >>> dates = pd.date_range('2024-01-01 09:30', periods=100, freq='5min')
    >>> data = pd.DataFrame({
    ...     'high': [102]*100,
    ...     'low': [100]*100,
    ...     'close': [101]*100
    ... }, index=dates)
    >>> result = atr_intraday(data)
    >>> # Returns DataFrame with time-of-day adjusted ATR
  """
  # Calculate true range first
  highs = np.asarray(data["high"].values, dtype=np.float64)
  lows = np.asarray(data["low"].values, dtype=np.float64)
  closes = np.asarray(data["close"].values, dtype=np.float64)

  true_ranges = compute_true_range_numba(highs, lows, closes)

  # Add true_range to dataframe for intraday aggregation
  data_with_tr = data.copy()
  data_with_tr["true_range"] = true_ranges

  # Get historical average true range for each time slot
  avg_tr_by_time = intraday_aggregate(
    data_with_tr["true_range"],
    agg_func="mean",
    lookback_days=lookback_days,
    min_samples=min_samples,
  )

  # Create result dataframe
  data_copy = data.copy()
  data_copy["true_range"] = true_ranges
  data_copy["atr_intraday"] = avg_tr_by_time

  return data_copy

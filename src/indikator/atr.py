"""Average True Range (ATR) indicator module.

This module provides ATR calculation, a volatility indicator that measures
the average range of price movement. Essential for position sizing and
stop-loss placement.
"""

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
  from numpy.typing import NDArray

from datawarden import (
  Columns,
  Datetime,
  Finite,
  Index,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Ge, Hyper, configurable
import numpy as np
import pandas as pd

from indikator._constants import DEFAULT_MIN_SAMPLES
from indikator._results import IndicatorResult
from indikator.numba.atr import (
  compute_atr_numba,
  compute_true_range_numba,
  compute_true_range_numba_2d,
)
from indikator.utils import to_numpy


@configurable
@validate
def atr(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 14,
) -> IndicatorResult:
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
  ATR measures market volatility. It decomposes the entire range of an asset
  for that period.

  Formula:
  TR = Max(High-Low, |High-PrevClose|, |Low-PrevClose|)
  ATR = EMA(TR, period)  (Wilder's Smoothing)

  Interpretation:
  - High ATR: High volatility (big moves)
  - Low ATR: Low volatility (consolidation)
  - Rising ATR: Volatility increasing (possible trend reversal/breakout)
  - ATR is NOT directional

  Features:
  - Numba-optimized for performance
  - Standard 14-period default (Wilder's original)

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    IndicatorResult(index, atr)
  """
  # Convert to numpy for Numba
  high_arr = to_numpy(high)
  low_arr = to_numpy(low)
  close_arr = to_numpy(close)

  # Calculate ATR using Numba-optimized function
  atr_values = compute_atr_numba(high_arr, low_arr, close_arr, period)

  return IndicatorResult(data_index=high.index, value=atr_values, name="atr")


@configurable
@validate
def atr_intraday(
  data: Validated[
    pd.DataFrame,
    Columns(["high", "low", "close"]),
    Finite,
    Index(Datetime),
    NotEmpty,
  ],
  lookback_days: Hyper[int] | None = None,
  min_samples: Hyper[int, Ge[2]] = DEFAULT_MIN_SAMPLES,
) -> IndicatorResult:
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
      Series with time-of-day adjusted ATR values (NaN until min_samples met per time slot)

    Raises:
      ValueError: If required columns missing or index is not DatetimeIndex

    Example:
      >>> import pandas as pd
  from indikator.utils import to_numpy
      >>> dates = pd.date_range('2024-01-01 09:30', periods=100, freq='5min')
      >>> data = pd.DataFrame({
      ...     'high': [102]*100,
      ...     'low': [100]*100,
      ...     'close': [101]*100
      ... }, index=dates)
      >>> result = atr_intraday(data)
      >>> # Returns DataFrame with time-of-day adjusted ATR
  """
  from indikator.numba.intraday import compute_intraday_mean_numba, time_to_key

  # Calculate true range first
  ohlc_values = cast(
    "NDArray[np.float64]",
    data[["high", "low", "close"]].to_numpy(dtype=np.float64, copy=False),
  )
  true_ranges = compute_true_range_numba_2d(ohlc_values)

  # Convert datetime index to time keys (seconds since midnight)
  dt_index = cast("pd.DatetimeIndex", data.index)

  # Apply lookback filter if specified
  if lookback_days is not None:
    cutoff_date = dt_index[-1] - pd.Timedelta(days=lookback_days)
    mask = dt_index >= cutoff_date
    tr_filtered = np.where(mask, true_ranges, np.nan)
  else:
    tr_filtered = true_ranges

  time_keys = time_to_key(dt_index)

  # Compute intraday mean using Numba kernel
  vals = compute_intraday_mean_numba(tr_filtered, time_keys, min_samples)

  return IndicatorResult(data_index=data.index, value=vals, name="atr_intraday")


@configurable
@validate
def trange(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> IndicatorResult:
  """Calculate True Range (TRANGE).

  The True Range is the greatest of:
  - Current high - current low
  - |Current high - previous close|
  - |Current low - previous close|

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.

  Returns:
    TRangeResult(index, trange)
  """
  # Convert to numpy for Numba
  high_arr = to_numpy(high)
  low_arr = to_numpy(low)
  close_arr = to_numpy(close)

  # Calculate TR using Numba-optimized function
  tr_values = compute_true_range_numba(high_arr, low_arr, close_arr)

  return IndicatorResult(data_index=high.index, value=tr_values, name="trange")

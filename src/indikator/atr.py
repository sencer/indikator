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

from indikator._atr_numba import (
  compute_atr_numba,
  compute_true_range_numba,
  compute_true_range_numba_2d,
)
from indikator._constants import DEFAULT_MIN_SAMPLES
from indikator._intraday import intraday_aggregate
from indikator._results import ATRIntradayResult, ATRResult, TRANGEResult
from indikator.utils import to_numpy


@configurable
@validate
def atr(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 14,
) -> ATRResult:
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
    ATRResult(index, atr)
  """
  # Convert to numpy for Numba
  high_arr = to_numpy(high)
  low_arr = to_numpy(low)
  close_arr = to_numpy(close)

  # Calculate ATR using Numba-optimized function
  atr_values = compute_atr_numba(high_arr, low_arr, close_arr, period)

  return ATRResult(data_index=high.index, atr=atr_values)


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
) -> ATRIntradayResult:
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
  # Calculate true range first
  # Optimization: Use bulk 2D access for speed since we have a DataFrame
  # We need High, Low, Close columns. We can extract them as a block if they are adjacent,
  # but safely we should select them.
  # data[["high", "low", "close"]] guarantees order 0=H, 1=L, 2=C
  ohlc_values = cast(
    "NDArray[np.float64]",
    data[["high", "low", "close"]].to_numpy(dtype=np.float64, copy=False),
  )

  true_ranges = compute_true_range_numba_2d(ohlc_values)

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

  # Return only the indicator (minimal return philosophy)
  # avg_tr_by_time is IntradaySeriesResult
  vals = avg_tr_by_time.values
  return ATRIntradayResult(data_index=data.index, atr_intraday=vals)


@configurable
@validate
def trange(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
) -> TRANGEResult:
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

  return TRANGEResult(data_index=high.index, trange=tr_values)

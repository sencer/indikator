"""Volume-Weighted Average Price (VWAP) indicator module.

This module provides VWAP calculation, a key intraday benchmark used by
institutional traders for execution quality and price reference.
"""

from typing import TYPE_CHECKING, Literal, cast

from hipr import configurable
import numpy as np
import pandas as pd
from pdval import (
  Datetime,
  Ge as GeValidator,
  HasColumns,
  Index,
  Validated,
  validated,
)

if TYPE_CHECKING:
  from numpy.typing import NDArray

from indikator._vwap_numba import compute_anchored_vwap_numba, compute_vwap_numba


@configurable
@validated
def vwap(
  data: Validated[
    pd.DataFrame,
    HasColumns[Literal["high", "low", "close", "volume"]],
    Index[Datetime],
    GeValidator[Literal["high", "low"]],
  ],
  session_freq: Literal["D", "W", "ME"] = "D",
) -> pd.DataFrame:
  """Calculate Volume-Weighted Average Price (VWAP).

  VWAP is the ratio of cumulative (price * volume) to cumulative volume,
  typically reset at the beginning of each trading session. It represents
  the average price weighted by volume.

  VWAP = Sum(Typical Price * Volume) / Sum(Volume)
  where Typical Price = (High + Low + Close) / 3

  Institutional traders use VWAP as:
  - Execution benchmark (am I getting better/worse than VWAP?)
  - Support/resistance level (price tends to revert to VWAP)
  - Trend indicator (price above VWAP = bullish, below = bearish)
  - Entry/exit signal (crossing VWAP can indicate trend changes)

  Features:
  - Numba-optimized for performance
  - Configurable session period (daily, weekly, monthly)
  - Handles missing volume gracefully
  - Returns both VWAP and typical price

  Args:
    data: OHLCV DataFrame with DatetimeIndex
    session_freq: Session reset frequency ('D'=daily, 'W'=weekly, 'ME'=month-end)

  Returns:
    DataFrame with 'vwap' and 'typical_price' columns added

  Raises:
    ValueError: If required columns missing or index not DatetimeIndex

  Example:
    >>> import pandas as pd
    >>> dates = pd.date_range('2024-01-01 09:30', periods=10, freq='5min')
    >>> data = pd.DataFrame({
    ...     'high': [102, 104, 103, 106, 108, 107, 109, 108, 110, 112],
    ...     'low': [100, 101, 100, 103, 105, 104, 106, 105, 107, 109],
    ...     'close': [101, 103, 102, 105, 107, 106, 108, 107, 109, 111],
    ...     'volume': [1000]*10
    ... }, index=dates)
    >>> result = vwap(data)
    >>> # Returns DataFrame with VWAP column
  """

  # Calculate typical price (H + L + C) / 3
  typical_price = (data["high"] + data["low"] + data["close"]) / 3.0

  # Create reset mask based on session frequency
  # Reset at the start of each new period
  dates = pd.Series(data.index, index=data.index)

  if session_freq == "D":
    period_start = dates.dt.normalize()
  elif session_freq == "W":
    period_start = dates.dt.to_period("W").dt.start_time
  elif session_freq == "ME":
    period_start = dates.dt.to_period("M").dt.start_time
  else:
    raise ValueError(f"Invalid session_freq: {session_freq}")

  # Reset mask is True where period changes (vectorized)
  reset_mask = np.asarray(period_start != period_start.shift(1))
  reset_mask[0] = True  # Always reset at first bar

  # Convert to numpy arrays for Numba
  typical_prices = np.asarray(typical_price.values, dtype=np.float64)
  volumes = np.asarray(data["volume"].values, dtype=np.float64)

  # Calculate VWAP using Numba-optimized function
  vwap_values = compute_vwap_numba(typical_prices, volumes, reset_mask)

  # Create result dataframe
  data_copy = data.copy()
  data_copy["typical_price"] = typical_price
  data_copy["vwap"] = vwap_values

  return data_copy


@configurable
@validated
def vwap_anchored(
  data: Validated[
    pd.DataFrame,
    HasColumns[Literal["high", "low", "close", "volume"]],
  ],
  anchor_index: int | None = None,
  anchor_datetime: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
  """Calculate Anchored VWAP from a specific point in time.

  Anchored VWAP calculates VWAP starting from a specific bar forward,
  rather than resetting at session boundaries. This is useful for:
  - Anchoring to significant events (earnings, news, pivots)
  - Tracking institutional positioning from specific entry points
  - Measuring average fill price from a particular time
  - Swing trading support/resistance from key levels

  Common anchor points:
  - Earnings announcements
  - Market structure breaks (new high/low)
  - Major news events
  - Session open/close
  - Previous day high/low

  Features:
  - Anchor by index position or datetime
  - No session resets (continuous from anchor)
  - Returns NaN for all bars before anchor
  - Numba-optimized for performance

  Args:
    data: OHLCV DataFrame
    anchor_index: Bar index to start VWAP calculation (0-based)
    anchor_datetime: Datetime to start VWAP (alternative to anchor_index)

  Returns:
    DataFrame with 'vwap_anchored' and 'typical_price' columns added

  Raises:
    ValueError: If neither or both anchor parameters provided, or anchor not found

  Example:
    >>> import pandas as pd
    >>> dates = pd.date_range('2024-01-01', periods=10, freq='D')
    >>> data = pd.DataFrame({
    ...     'high': [102, 104, 103, 106, 108, 107, 109, 108, 110, 112],
    ...     'low': [100, 101, 100, 103, 105, 104, 106, 105, 107, 109],
    ...     'close': [101, 103, 102, 105, 107, 106, 108, 107, 109, 111],
    ...     'volume': [1000]*10
    ... }, index=dates)
    >>> # Anchor VWAP from bar 3 (representing a breakout)
    >>> result = vwap_anchored(data, anchor_index=3)
    >>> # Or anchor from specific date
    >>> result = vwap_anchored(data, anchor_datetime='2024-01-04')
  """
  # Validate anchor parameters
  if anchor_index is None and anchor_datetime is None:
    raise ValueError("Must provide either anchor_index or anchor_datetime")
  if anchor_index is not None and anchor_datetime is not None:
    raise ValueError("Cannot provide both anchor_index and anchor_datetime")

  # Resolve anchor_index from datetime if needed
  if anchor_datetime is not None:
    if isinstance(anchor_datetime, str):
      anchor_datetime = pd.Timestamp(anchor_datetime)

    # Find the index position for the anchor datetime
    if isinstance(data.index, pd.DatetimeIndex):
      # Find nearest datetime (or exact match)
      try:
        indexer_result = cast(
          "NDArray[np.intp]",
          data.index.get_indexer(  # pyright: ignore[reportUnknownMemberType]
            [anchor_datetime], method="nearest"
          ),
        )
        anchor_index = int(indexer_result[0])  # pyright: ignore[reportAny]
      except Exception as e:
        raise ValueError(
          f"Could not find anchor_datetime {anchor_datetime} in index"
        ) from e
    else:
      raise ValueError(
        "anchor_datetime requires DatetimeIndex, but index is not DatetimeIndex"
      )

  # Validate anchor_index range
  if anchor_index is None:
    # This should be unreachable due to earlier check "if anchor_index is None and anchor_datetime is None"
    # and anchor_datetime resolution logic, but explicit check helps type checker
    raise ValueError("anchor_index resolution failed")

  if anchor_index < 0 or anchor_index >= len(data):
    raise ValueError(f"anchor_index {anchor_index} out of range [0, {len(data) - 1}]")

  # Calculate typical price (H + L + C) / 3
  typical_price = (data["high"] + data["low"] + data["close"]) / 3.0

  # Convert to numpy arrays for Numba
  typical_prices = np.asarray(typical_price.values, dtype=np.float64)
  volumes = np.asarray(data["volume"].values, dtype=np.float64)

  # Calculate anchored VWAP using Numba-optimized function
  vwap_values = compute_anchored_vwap_numba(typical_prices, volumes, anchor_index)

  # Create result dataframe
  data_copy = data.copy()
  data_copy["typical_price"] = typical_price
  data_copy["vwap_anchored"] = vwap_values

  return data_copy

"""Volume-Weighted Average Price (VWAP) indicator module.

This module provides VWAP calculation, a key intraday benchmark used by
institutional traders for execution quality and price reference.
"""

from typing import TYPE_CHECKING, cast

from datawarden import (
  Columns,
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Hyper, configurable
import numpy as np
import pandas as pd

from indikator.utils import to_numpy

if TYPE_CHECKING:
  from numpy.typing import NDArray

from indikator._results import IndicatorResult
from indikator.numba.vwap import (
  compute_anchored_vwap_numba,
  compute_vwap_numba,
  compute_vwap_parallel_numba,
)

PARALLEL_THRESHOLD = 5


@configurable
@validate
def vwap(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  volume: Validated[pd.Series[float], Finite, NotEmpty],
  anchor: Hyper[str | pd.Timedelta | int] = "D",
) -> IndicatorResult:
  """Calculate Volume Weighted Average Price (VWAP).

  VWAP is a trading benchmark that gives the average price a security has
  traded at throughout the day, based on both volume and price.

  Formula:
  Typical Price = (High + Low + Close) / 3
  VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)

  Reset:
  The cumulative sums reset based on the anchor period (e.g., daily 'D').

  Interpretation:
  - Price > VWAP: Bullish sentiment (buyers are in control)
  - Price < VWAP: Bearish sentiment (sellers are in control)
  - VWAP acts as dynamic support/resistance
  - Institutions use VWAP to execute large orders without moving market

  Features:
  - Numba-optimized for performance
  - Flexible anchoring (Time-based or Bar-count based)
  - Standard 'D' (daily) anchor default

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    volume: Volume Series.
    anchor: Reset anchor (e.g. 'D', 'W', '1h') or int (bars). Default 'D'.

  Returns:
    IndicatorResult(index, vwap)
  """
  # Align all inputs
  # Note: Validator ensures equal length and index alignment

  # Calculate reset_mask
  if isinstance(anchor, int):
    # Reset every N bars
    # Create boolean mask where index % anchor == 0
    # Or just start with False and set True at indices
    n = len(high)
    reset_mask = np.zeros(n, dtype=np.bool_)
    reset_mask[::anchor] = True
  else:
    # Time-based anchor
    if not isinstance(high.index, pd.DatetimeIndex):
      raise ValueError("Index must be DatetimeIndex for time-based anchor")

    grouper = high.index.to_period(anchor)  # type: ignore
    # Reset where group changes
    reset_mask = np.concatenate(([True], grouper[1:] != grouper[:-1]))

  # Convert to numpy for Numba
  high_arr = to_numpy(high)
  low_arr = to_numpy(low)
  close_arr = to_numpy(close)
  vol_arr = to_numpy(volume)

  # Calculate VWAP using Numba-optimized function
  # If we have multiple sessions, use the parallel version for speedup
  reset_indices = np.where(reset_mask)[0].astype(np.int64)
  if len(reset_indices) > PARALLEL_THRESHOLD:  # Threshold for parallel gain
    vwap_values = compute_vwap_parallel_numba(
      high_arr, low_arr, close_arr, vol_arr, reset_indices
    )
  else:
    vwap_values = compute_vwap_numba(high_arr, low_arr, close_arr, vol_arr, reset_mask)

  return IndicatorResult(data_index=high.index, value=vwap_values, name="vwap")


@configurable
@validate
def vwap_anchored(
  data: Validated[
    pd.DataFrame,
    Columns(["high", "low", "close", "volume"]),
    Finite,
    NotEmpty,
  ],
  anchor_index: Hyper[int] | None = None,
  anchor_datetime: Hyper[pd.Timestamp | str] | None = None,
) -> IndicatorResult:
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
  from indikator.utils import to_numpy
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
        anchor_index = int(indexer_result[0])
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

  # Return only the indicator (minimal return philosophy)
  return IndicatorResult(data_index=data.index, value=vwap_values, name="vwap_anchored")

"""ZigZag Legs indicator module.

This module provides zigzag leg counting with Elliott Wave-style market
structure tracking. It distinguishes between corrections (temporary moves
against the trend) and trend changes (structure breaks).
"""

from typing import Literal

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Gt, Hyper, configurable
import numpy as np
import pandas as pd

from indikator._results import ZigzagLegsResult
from indikator.numba.legs import compute_zigzag_numba
from indikator.utils import to_numpy


@configurable
@validate
def legs(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  deviation: Hyper[float, Gt[0.0]] = 0.05,
  method: Hyper[Literal["percentage", "absolute"]] = "percentage",
) -> ZigzagLegsResult:
  """Calculate ZigZag legs.

  Identifies swing highs and lows that exceed a minimum deviation threshold.
  Used to filter out noise and identify significant market moves.

  Features:
  - Numba-optimized
  - "Percentage" or "Absolute" deviation modes
  - Returns structured legs data (start/end price, direction)

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    deviation: Minimum change to qualify as a new leg (0.05 = 5%).
    method: 'percentage' or 'absolute'.

  Returns:

  Raises:
    ValueError: If data contains NaN or infinite values

  Example:
    >>> import pandas as pd
    >>> # Bullish structure: stays positive during corrections
    >>> data = pd.Series([100, 110, 105, 115])  # Up, down (correction), up
    >>> result = zigzag_legs(data, threshold=0.03, confirmation_bars=0)
    >>> # Output: [0, 1, 1, 1] - stays positive (bullish structure)
    >>>
    >>> # Structure break: negative when breaking previous low
    >>> data2 = pd.Series([100, 110, 105, 115, 100, 95])  # Breaks below 105
    >>> result2 = zigzag_legs(data2, threshold=0.03, confirmation_bars=0)
    >>> # After price breaks previous low at 105, structure changes to bearish
  """
  # We use valid data but need to ensure types for Numba
  # Convert to numpy arrays for Numba
  # Use high/low for standard Zigzag
  high_arr = to_numpy(high)
  low_arr = to_numpy(low)
  close_arr = to_numpy(close)

  percentage_mode = method == "percentage"

  directions, start_indices, end_indices, start_prices, end_prices = (
    compute_zigzag_numba(
      high_arr,
      low_arr,
      close_arr,
      deviation,
      percentage_mode,
    )
  )

  # Convert integer indices to the original index values if possible?
  # The Result should probably store integer indices and the pandas Index separately.
  # ZigzagLegsResult definition in _results.py includes:
  # - index: pd.Index
  # - direction: NDArray
  # - start_index: int array
  # - end_index: int array
  # - start_price: float array
  # - end_price: float array

  # Calculate pct_change for result
  # Avoid division by zero
  pct_change = np.zeros_like(start_prices)
  valid = start_prices != 0
  pct_change[valid] = (end_prices[valid] - start_prices[valid]) / start_prices[valid]

  return ZigzagLegsResult(
    data_index=pd.RangeIndex(len(directions)),  # Use RangeIndex for legs list
    direction=directions,
    start_price=start_prices,
    end_price=end_prices,
    start_idx=start_indices,
    end_idx=end_indices,
    pct_change=pct_change,
  )

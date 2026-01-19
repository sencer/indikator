"""CMO (Chande Momentum Oscillator) indicator module.

This module provides CMO calculation, a momentum oscillator that measures
the difference between sum of gains and losses over a period.
"""

from typing import TYPE_CHECKING, cast

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Ge, Hyper, configurable
import numpy as np
import pandas as pd

if TYPE_CHECKING:
  from numpy.typing import NDArray

from indikator._cmo_numba import compute_cmo_numba
from indikator._results import CMOResult


@configurable
@validate
def cmo(
  data: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
) -> CMOResult:
  """Calculate Chande Momentum Oscillator (CMO).

  CMO measures the momentum of price changes. It oscillates between -100
  and +100, making it useful for identifying overbought/oversold conditions.

  Formula:
  CMO = 100 * (sum_gains - sum_losses) / (sum_gains + sum_losses)

  Interpretation:
  - CMO > 50: Overbought (potential reversal down)
  - CMO < -50: Oversold (potential reversal up)
  - CMO = 0: Equal gains and losses
  - Crossing zero: Momentum shift

  Unlike RSI which divides sum_gains by sum_losses, CMO uses their
  difference divided by their sum, giving a true center at zero.

  Features:
  - Numba-optimized for performance
  - O(n) sliding window algorithm
  - Range: -100 to +100 (centered at 0)

  Args:
    data: Input Series (typically closing prices)
    period: Lookback period (default: 14)

  Returns:
    CMOResult(index, cmo)

  Raises:
    ValueError: If data contains NaN/Inf

  Example:
    >>> import pandas as pd
    >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109] * 3)
    >>> result = cmo(prices, period=5).to_pandas()
    >>> # Returns CMO values
  """
  # Convert to numpy for Numba
  values = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  # Calculate CMO using Numba-optimized function
  cmo_values = compute_cmo_numba(values, period)

  return CMOResult(index=data.index, cmo=cmo_values)

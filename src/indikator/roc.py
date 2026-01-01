"""ROC (Rate of Change) indicator module.

This module provides ROC calculation, a momentum oscillator that measures
the percentage change between the current price and the price n periods ago.
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

from indikator._roc_numba import compute_roc_numba


@configurable
@validate
def roc(
  data: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 10,
) -> pd.Series:
  """Calculate Rate of Change (ROC).

  ROC is a momentum oscillator that measures the percentage change between
  the current price and the price n periods ago.

  Formula:
  ROC = ((Price - Price_n_periods_ago) / Price_n_periods_ago) * 100

  Interpretation:
  - ROC > 0: Price is higher than n periods ago (bullish)
  - ROC < 0: Price is lower than n periods ago (bearish)
  - ROC crossing 0: Momentum shift
  - Extreme readings: Potential reversal

  Common strategies:
  - Buy on positive ROC (momentum confirmation)
  - Sell on negative ROC
  - Look for divergences between price and ROC

  Features:
  - Numba-optimized for performance
  - Standard 10-period default
  - Simple percentage-based calculation

  Args:
    data: Input Series.
    period: Lookback period (default: 10)

  Returns:
    Series with ROC values (percentage)

  Raises:
    ValueError: If data contains NaN/Inf

  Example:
    >>> import pandas as pd
    >>> prices = pd.Series([100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120])
    >>> result = roc(prices, period=5)
  """
  # Convert to numpy for Numba
  values = cast(
    "NDArray[np.float64]",
    data.values.astype(np.float64),  # pyright: ignore[reportUnknownMemberType]
  )

  # Calculate ROC using Numba-optimized function
  roc_values = compute_roc_numba(values, period)

  return pd.Series(roc_values, index=data.index, name="roc")

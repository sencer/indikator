"""Kaufman Adaptive Moving Average (KAMA) indicator module.

KAMA adapts its responsiveness based on market efficiency.
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

from indikator._kama_numba import compute_kama_numba
from indikator._results import KAMAResult


@configurable
@validate
def kama(
  data: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 10,
  fast_period: Hyper[int, Ge[2]] = 2,
  slow_period: Hyper[int, Ge[2]] = 30,
) -> KAMAResult:
  """Calculate Kaufman Adaptive Moving Average (KAMA).

  KAMA adjusts its smoothing constant based on market efficiency:
  - In trending markets (high efficiency): responds quickly
  - In choppy markets (low efficiency): smooths more

  Formula:
  ER = |Price - Price_n_ago| / sum(|daily changes|)
  SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
  KAMA = KAMA_prev + SC * (Price - KAMA_prev)

  Interpretation:
  - KAMA slope indicates trend direction
  - Flat KAMA suggests ranging market
  - Price crossing KAMA can signal trend changes

  Features:
  - O(1) rolling volatility calculation
  - Adaptive smoothing responds to market conditions

  Args:
    data: Input price Series (typically close prices)
    period: Efficiency ratio lookback (default: 10)
    fast_period: Fast smoothing period (default: 2)
    slow_period: Slow smoothing period (default: 30)

  Returns:
    KAMAResult with adaptive moving average values

  Example:
    >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108])
    >>> result = kama(prices, period=10)
  """
  values = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  kama_values = compute_kama_numba(values, period, fast_period, slow_period)

  return KAMAResult(index=data.index, kama=kama_values)

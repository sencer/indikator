"""Momentum (MOM) indicator module."""

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

from indikator._mom_numba import compute_mom_numba
from indikator._results import MOMResult


@configurable
@validate
def mom(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 10,
) -> MOMResult:
  """Calculate Momentum (MOM).

  Momentum measures the absolute price change over a specified period.
  Unlike ROC which shows percentage change, MOM shows raw price difference.

  Formula:
  MOM = Price(t) - Price(t - period)

  Interpretation:
  - Positive MOM: Price ascending
  - Negative MOM: Price descending
  - Zero crossing: Potential trend change
  - Divergence from price: Potential reversal

  Common uses:
  - Trend confirmation
  - Overbought/oversold detection
  - Divergence analysis
  - Leading indicator for price reversals

  Features:
  - Numba-optimized with parallel execution
  - Simple and fast O(N) calculation

  Args:
    data: Input price Series (typically close prices)
    period: Lookback period (default: 10)

  Returns:
    MOMResult with momentum values

  Example:
    >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108])
    >>> result = mom(prices, period=3)
  """
  # Convert to numpy for Numba
  values = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  mom_values = compute_mom_numba(values, period)

  return MOMResult(data_index=data.index, mom=mom_values)

"""Momentum (MOM) indicator module."""

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Ge, Hyper, configurable
import pandas as pd

from indikator._results import IndicatorResult
from indikator.numba.mom import compute_mom_numba
from indikator.utils import to_numpy


@configurable
@validate
def mom(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 10,
) -> IndicatorResult:
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
    IndicatorResult with momentum values

  Example:
    >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108])
    >>> result = mom(prices, period=3)
  """
  # Convert to numpy for Numba
  values = to_numpy(data)

  mom_values = compute_mom_numba(values, period)

  return IndicatorResult(data_index=data.index, value=mom_values, name="mom")

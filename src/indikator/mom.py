"""Momentum (MOM) indicator module.

Measures the rate of change in price over a specified period.
"""

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Ge, Hyper, configurable
import numpy as np
import pandas as pd

from indikator._mom_numba import compute_mom_numba
from indikator._results import MOMResult


@configurable
@validate
def mom(
  data: Validated[pd.Series, Finite, NotEmpty],
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
  # Optimization: Access .values directly.
  # Validation ensures Finite/Numeric data, so strict casting checks can be relaxed for speed.
  # If data is int, np.subtract(..., out=float_arr) handles it correctly.
  values = data.values

  # Ensure we have a numpy array (e.g. if Series was wrapping something else, though unlikely)
  # and if it's object type (rare given validation), let numpy handle it in computation or fail fast.
  if not isinstance(values, np.ndarray):
    values = data.to_numpy(dtype=np.float64, copy=False)

  mom_values = compute_mom_numba(values, period)

  return MOMResult(data.index, mom_values)

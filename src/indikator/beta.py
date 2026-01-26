"""Beta coefficient indicator module.

This module provides Numba-optimized implementations of:
- beta_statistical: Beta on raw inputs (flexible)
- beta: Beta on 1-period returns (matches TA-Lib)
"""

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Ge, Hyper, configurable
import pandas as pd

from indikator._results import IndicatorResult
from indikator.numba.correlation import (
  compute_beta_fused_rocp_numba,
  compute_beta_numba,
)
from indikator.utils import to_numpy


@configurable
@validate
def beta_statistical(
  x: Validated[pd.Series[float], Finite, NotEmpty],
  y: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 5,
) -> IndicatorResult:
  """Calculate rolling BETA coefficient on RAW INPUTS.

  This is the pure statistical calculation of Beta:
    BETA = cov(X, Y) / var(X)

  This function does NOT transform inputs into returns. It calculates
  beta directly on the provided x and y series.

  Use this if you have already calculated returns or want beta of prices.

  Args:
    x: Independent variable (e.g., market returns)
    y: Dependent variable (e.g., stock returns)
    period: Rolling window size (default: 5)

  Returns:
    IndicatorResult(index, beta)
  """
  x_arr = to_numpy(x)
  y_arr = to_numpy(y)

  result = compute_beta_numba(x_arr, y_arr, period)

  return IndicatorResult(data_index=x.index, value=result, name="beta")


@configurable
@validate
def beta(
  x: Validated[pd.Series[float], Finite, NotEmpty],
  y: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 5,
) -> IndicatorResult:
  """Calculate rolling BETA coefficient (TA-Lib compatible).

  Matches TA-Lib's behavior: automatically calculates 1-period
  percentage change (returns) for both inputs before calculating Beta.

  BETA = cov(rocp(x), rocp(y)) / var(rocp(x))

  Performance Note:
  Uses a FUSED kernel that calculates ROCP on the fly within the rolling loop,
  avoiding intermediate array allocations.

  Args:
    x: Independent variable price series (e.g., market index)
    y: Dependent variable price series (e.g., stock price)
    period: Rolling window size (default: 5)

  Returns:
    IndicatorResult(index, beta)
  """
  x_arr = to_numpy(x)
  y_arr = to_numpy(y)

  # Use fused kernel for optimal performance
  result = compute_beta_fused_rocp_numba(x_arr, y_arr, period)

  return IndicatorResult(data_index=x.index, value=result, name="beta")

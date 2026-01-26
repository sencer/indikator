"""ROC (Rate of Change) family of indicators.

This module provides ROC, ROCP, ROCR, and ROCR100 calculations.
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
from indikator.numba.roc import (
  compute_roc_numba,
  compute_rocp_numba,
  compute_rocr100_numba,
  compute_rocr_numba,
)
from indikator.utils import to_numpy


@configurable
@validate
def roc(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 10,
) -> IndicatorResult:
  """Calculate Rate of Change (ROC).

  ROC = ((Price - Prev) / Prev) * 100
  """
  values = to_numpy(data)
  result = compute_roc_numba(values, period)
  return IndicatorResult(data_index=data.index, value=result, name="roc")


@configurable
@validate
def rocp(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 10,
) -> IndicatorResult:
  """Calculate Rate of Change Percentage (ROCP).

  ROCP = (Price - Prev) / Prev
  """
  values = to_numpy(data)
  result = compute_rocp_numba(values, period)
  return IndicatorResult(data_index=data.index, value=result, name="rocp")


@configurable
@validate
def rocr(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 10,
) -> IndicatorResult:
  """Calculate Rate of Change Ratio (ROCR).

  ROCR = Price / Prev
  """
  values = to_numpy(data)
  result = compute_rocr_numba(values, period)
  return IndicatorResult(data_index=data.index, value=result, name="rocr")


@configurable
@validate
def rocr100(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 10,
) -> IndicatorResult:
  """Calculate Rate of Change Ratio 100 Scale (ROCR100).

  ROCR100 = (Price / Prev) * 100
  """
  values = to_numpy(data)
  result = compute_rocr100_numba(values, period)
  return IndicatorResult(data_index=data.index, value=result, name="rocr100")

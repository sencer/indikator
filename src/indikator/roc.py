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

from indikator._results import ROCPResult, ROCR100Result, ROCResult, ROCRResult
from indikator._roc_numba import (
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
) -> ROCResult:
  """Calculate Rate of Change (ROC).

  ROC = ((Price - Prev) / Prev) * 100
  """
  values = to_numpy(data)
  result = compute_roc_numba(values, period)
  return ROCResult(data_index=data.index, roc=result)


@configurable
@validate
def rocp(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 10,
) -> ROCPResult:
  """Calculate Rate of Change Percentage (ROCP).

  ROCP = (Price - Prev) / Prev
  """
  values = to_numpy(data)
  result = compute_rocp_numba(values, period)
  return ROCPResult(data_index=data.index, rocp=result)


@configurable
@validate
def rocr(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 10,
) -> ROCRResult:
  """Calculate Rate of Change Ratio (ROCR).

  ROCR = Price / Prev
  """
  values = to_numpy(data)
  result = compute_rocr_numba(values, period)
  return ROCRResult(data_index=data.index, rocr=result)


@configurable
@validate
def rocr100(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 10,
) -> ROCR100Result:
  """Calculate Rate of Change Ratio 100 Scale (ROCR100).

  ROCR100 = (Price / Prev) * 100
  """
  values = to_numpy(data)
  result = compute_rocr100_numba(values, period)
  return ROCR100Result(data_index=data.index, rocr100=result)

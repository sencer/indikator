"""ROC (Rate of Change) family of indicators.

This module provides ROC, ROCP, ROCR, and ROCR100 calculations.
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

from indikator._results import ROCPResult, ROCR100Result, ROCResult, ROCRResult
from indikator._roc_numba import (
  compute_roc_numba,
  compute_rocp_numba,
  compute_rocr100_numba,
  compute_rocr_numba,
)


@configurable
@validate
def roc(
  data: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 10,
) -> ROCResult:
  """Calculate Rate of Change (ROC).

  ROC = ((Price - Prev) / Prev) * 100
  """
  values = cast("NDArray[np.float64]", data.to_numpy(dtype=np.float64, copy=False))
  result = compute_roc_numba(values, period)
  return ROCResult(index=data.index, roc=result)


@configurable
@validate
def rocp(
  data: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 10,
) -> ROCPResult:
  """Calculate Rate of Change Percentage (ROCP).

  ROCP = (Price - Prev) / Prev
  """
  values = cast("NDArray[np.float64]", data.to_numpy(dtype=np.float64, copy=False))
  result = compute_rocp_numba(values, period)
  return ROCPResult(index=data.index, rocp=result)


@configurable
@validate
def rocr(
  data: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 10,
) -> ROCRResult:
  """Calculate Rate of Change Ratio (ROCR).

  ROCR = Price / Prev
  """
  values = cast("NDArray[np.float64]", data.to_numpy(dtype=np.float64, copy=False))
  result = compute_rocr_numba(values, period)
  return ROCRResult(index=data.index, rocr=result)


@configurable
@validate
def rocr100(
  data: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[1]] = 10,
) -> ROCR100Result:
  """Calculate Rate of Change Ratio 100 Scale (ROCR100).

  ROCR100 = (Price / Prev) * 100
  """
  values = cast("NDArray[np.float64]", data.to_numpy(dtype=np.float64, copy=False))
  result = compute_rocr100_numba(values, period)
  return ROCR100Result(index=data.index, rocr100=result)

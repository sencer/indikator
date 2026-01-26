"""Correlation coefficient indicator module.

This module provides a Numba-optimized implementation of rolling CORREL,
the Pearson correlation coefficient.
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
from indikator.numba.correlation import compute_correl_numba
from indikator.utils import to_numpy


@configurable
@validate
def correl(
  x: Validated[pd.Series[float], Finite, NotEmpty],
  y: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 30,
) -> IndicatorResult:
  """Calculate rolling Pearson correlation coefficient.

  CORREL measures the linear relationship between two variables.
  Range: -1 to +1

  Interpretation:
  - CORREL = +1: Perfect positive correlation
  - CORREL = 0: No linear correlation
  - CORREL = -1: Perfect negative correlation

  Uses O(1) rolling update for efficiency.

  Args:
    x: First variable
    y: Second variable
    period: Rolling window size (default: 30)

  Returns:
    IndicatorResult(index, correl)
  """
  x_arr = to_numpy(x)
  y_arr = to_numpy(y)

  result = compute_correl_numba(x_arr, y_arr, period)

  return IndicatorResult(data_index=x.index, value=result, name="correl")

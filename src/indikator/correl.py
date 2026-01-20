"""Correlation coefficient indicator module.

This module provides a Numba-optimized implementation of rolling CORREL,
the Pearson correlation coefficient.
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

from indikator._correlation_numba import compute_correl_numba
from indikator._results import CORRELResult


@configurable
@validate
def correl(
  x: Validated[pd.Series, Finite, NotEmpty],
  y: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 30,
) -> CORRELResult:
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
    CORRELResult(index, correl)
  """
  x_arr = cast(
    "NDArray[np.float64]",
    x.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )
  y_arr = cast(
    "NDArray[np.float64]",
    y.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  result = compute_correl_numba(x_arr, y_arr, period)

  return CORRELResult(index=x.index, correl=result)

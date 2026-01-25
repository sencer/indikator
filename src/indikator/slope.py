"""Linear regression slope indicator module.

This module provides a Numba-optimized rolling slope calculation using
linear regression. 1,000-8,000x faster than using scipy.linregress.
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


from indikator._results import SlopeResult
from indikator._slope_numba import compute_slope_numba


@configurable
@validate
def slope(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  window: Hyper[int, Ge[2]] = 20,
) -> SlopeResult:
  """Calculate the slope of linear regression over a rolling window.

  The slope indicates the direction and steepness of the trend:
  - Positive slope: Uptrend
  - Negative slope: Downtrend
  - Near zero: Sideways/Consolidation

  Args:
    data: Series of prices (e.g., close prices)
    window: Rolling window size for regression

  Returns:
    SlopeResult(index, slope)

  Raises:
    ValueError: If validation fails
  """

  # Convert to numpy for Numba
  values = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  # Calculate slopes using Numba-optimized function
  slopes = compute_slope_numba(values, window)

  return SlopeResult(data_index=data.index, slope=slopes)

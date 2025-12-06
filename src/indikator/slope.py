"""Linear regression slope indicator module.

This module provides a Numba-optimized rolling slope calculation using
linear regression. 1,000-8,000x faster than using scipy.linregress.
"""
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from hipr import Ge, Hyper, configurable
import numpy as np
import pandas as pd
from pdval import (
    Finite,
    NonEmpty,
    NonNaN,
    Validated,
    validated,
)

from indikator._slope_numba import compute_slope_numba


@configurable
@validated
def slope(
    data: Validated[pd.Series, Finite, NonNaN, NonEmpty],
    window: Hyper[int, Ge[2]] = 20,
) -> pd.Series:
    """Calculate the slope of linear regression over a rolling window.

    The slope indicates the direction and steepness of the trend:
    - Positive slope: Uptrend
    - Negative slope: Downtrend
    - Near zero: Sideways/Consolidation

    Args:
      data: Series of prices (e.g., close prices)
      window: Rolling window size for regression

    Returns:
      Series with slope values

    Raises:
      ValueError: If validation fails
    """

    # Convert to numpy for Numba
    values = data.values.astype(np.float64)

    # Calculate slopes using Numba-optimized function
    slopes = compute_slope_numba(values, window)

    return pd.Series(slopes, index=data.index, name=data.name)

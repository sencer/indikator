"""On-Balance Volume (OBV) indicator module.

This module provides OBV calculation, a cumulative volume-based indicator
that relates volume to price change.
"""

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import configurable
import numpy as np
import pandas as pd

from indikator._obv_numba import compute_obv_numba
from indikator._results import OBVResult


@configurable
@validate
def obv(
  close: Validated[pd.Series, Finite, NotEmpty],
  volume: Validated[pd.Series, Finite, NotEmpty],
) -> OBVResult:
  """Calculate On Balance Volume (OBV).

  OBV measures buying and selling pressure as a cumulative indicator that
  adds volume on up days and subtracts volume on down days.

  Formula:
  If Close > Close_prev: OBV = OBV_prev + Volume
  If Close < Close_prev: OBV = OBV_prev - Volume
  If Close = Close_prev: OBV = OBV_prev

  Interpretation:
  - Volume accumulation: Rising OBV during consolidation = breakout coming

  Features:
  - Numba-optimized for performance
  - Cumulative calculation (no window parameter)
  - Handles flat price days (no volume change)
  - Simple and effective

  Args:
    data: OHLCV DataFrame with 'close' and 'volume' columns

  Returns:
    DataFrame with 'obv' column

  Raises:
    ValueError: If required columns missing or data contains NaN/Inf

  Example:
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'close': [100, 102, 101, 103, 105],
    ...     'volume': [1000, 1200, 900, 1500, 1100]
    ... })
    >>> result = obv(data)
    >>> # OBV = [1000, 2200, 1300, 2800, 3900]
  """

  # Convert to numpy arrays for Numba
  closes = np.asarray(close.values, dtype=np.float64)
  volumes = np.asarray(volume.values, dtype=np.float64)

  # Calculate OBV using Numba-optimized function
  obv_values = compute_obv_numba(closes, volumes)

  # Return only the indicator (minimal return philosophy)
  return OBVResult(index=close.index, obv=obv_values)

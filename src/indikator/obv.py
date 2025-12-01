"""On-Balance Volume (OBV) indicator module.

This module provides OBV calculation, a cumulative volume-based indicator
that relates volume to price change.
"""
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from typing import Literal

from hipr import configurable
import numpy as np
import pandas as pd
from pdval import (
    HasColumns,
    Validated,
    validated,
)

from indikator._obv_numba import compute_obv_numba


@configurable
@validated
def obv(
    data: Validated[pd.DataFrame, HasColumns[Literal["close", "volume"]]],
) -> pd.DataFrame:
    """Calculate On-Balance Volume (OBV).

    OBV is a cumulative indicator that adds volume on up days and subtracts
    volume on down days. It measures buying and selling pressure.

    Formula:
    - If close > previous close: OBV = OBV_previous + volume
    - If close < previous close: OBV = OBV_previous - volume
    - If close == previous close: OBV = OBV_previous

    Theory:
    - Volume precedes price (smart money accumulates before price rises)
    - Rising OBV with rising price = confirmed uptrend
    - Falling OBV with falling price = confirmed downtrend
    - OBV rising while price flat = accumulation (bullish)
    - OBV falling while price flat = distribution (bearish)

    Interpretation:
    - OBV trending up: Buying pressure increasing
    - OBV trending down: Selling pressure increasing
    - OBV divergence from price: Warning of potential reversal
    - OBV breakout before price: Early signal of price breakout

    Common strategies:
    - Trend confirmation: OBV should move with price trend
    - Divergence: OBV makes higher low while price makes lower low = bullish
    - Breakout confirmation: OBV breaks out with price = strong signal
    - Volume accumulation: Rising OBV during consolidation = breakout coming

    Features:
    - Numba-optimized for performance
    - Cumulative calculation (no window parameter)
    - Handles flat price days (no volume change)
    - Simple and effective

    Args:
      data: OHLCV DataFrame with 'close' and 'volume' columns

    Returns:
      DataFrame with 'obv' column added

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
    closes = np.asarray(data["close"].values, dtype=np.float64)
    volumes = np.asarray(data["volume"].values, dtype=np.float64)

    # Calculate OBV using Numba-optimized function
    obv_values = compute_obv_numba(closes, volumes)

    # Create result dataframe
    data_copy = data.copy()
    data_copy["obv"] = obv_values

    return data_copy

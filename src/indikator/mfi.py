"""Money Flow Index (MFI) indicator module.

This module provides MFI calculation, a momentum indicator that uses both
price and volume to measure buying and selling pressure.
"""

# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
from typing import Literal

from hipr import Ge, Gt, Hyper, configurable
import numpy as np
import pandas as pd
from pdval import (
    Ge as GeValidator,
    HasColumns,
    Validated,
    validated,
)

from indikator._mfi_numba import compute_mfi_numba


@configurable
@validated
def mfi(
    data: Validated[
        pd.DataFrame,
        HasColumns[Literal["high", "low", "close", "volume"]],
        GeValidator[Literal["high", "low"]],
    ],
    window: Hyper[int, Ge[2]] = 14,
    epsilon: Hyper[float, Gt[0.0]] = 1e-9,
) -> pd.DataFrame:
    """Calculate Money Flow Index (MFI).

    MFI is a momentum indicator that uses both price and volume to measure
    buying and selling pressure. It is also known as volume-weighted RSI.

    Formula:
    1. Typical Price = (High + Low + Close) / 3
    2. Money Flow = Typical Price * Volume
    3. Positive Money Flow = sum of MF when typical price increases
    4. Negative Money Flow = sum of MF when typical price decreases
    5. Money Flow Ratio = Positive Money Flow / Negative Money Flow
    6. MFI = 100 - (100 / (1 + Money Flow Ratio))

    Interpretation:
    - MFI > 80: Overbought (potential reversal down)
    - MFI < 20: Oversold (potential reversal up)
    - Divergence: MFI moves opposite to price (strong reversal signal)
    - Failure Swings: MFI crosses above 80 then below (sell) or below 20 then above (buy)

    Features:
    - Numba-optimized for performance
    - Uses typical price (H+L+C)/3
    - Handles division by zero with epsilon
    - 0-100 bounded range

    Args:
      data: OHLCV DataFrame
      window: Rolling window size (default: 14)
      epsilon: Small value to prevent division by zero

    Returns:
      DataFrame with 'mfi' column added

    Raises:
      ValueError: If required columns missing or data contains NaN/Inf

    Example:
      >>> import pandas as pd
      >>> data = pd.DataFrame({
      ...     'high': [10, 12, 11, 13, 15],
      ...     'low': [9, 10, 9, 11, 12],
      ...     'close': [9.5, 11, 10, 12, 14],
      ...     'volume': [100, 150, 120, 200, 180]
      ... })
      >>> result = mfi(data, window=3)
      >>> # Returns DataFrame with 'mfi' column
    """

    # Convert to numpy arrays for Numba
    highs = np.asarray(data["high"].values, dtype=np.float64)
    lows = np.asarray(data["low"].values, dtype=np.float64)
    closes = np.asarray(data["close"].values, dtype=np.float64)
    volumes = np.asarray(data["volume"].values, dtype=np.float64)

    # Calculate typical price
    typical_prices = (highs + lows + closes) / 3.0

    # Calculate MFI using Numba-optimized function
    mfi_values = compute_mfi_numba(typical_prices, volumes, window, epsilon)

    # Create result dataframe
    data_copy = data.copy()
    data_copy["typical_price"] = typical_prices
    data_copy["mfi"] = mfi_values

    return data_copy

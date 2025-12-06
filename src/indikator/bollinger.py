"""Bollinger Bands indicator module.

This module provides Bollinger Bands calculation, a volatility indicator
consisting of a moving average with upper and lower bands.
"""

# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from hipr import Ge, Gt, Hyper, configurable
import numpy as np
import pandas as pd
from pdval import (
    Finite,
    NonEmpty,
    Validated,
    validated,
)


@configurable
@validated
def bollinger_bands(
    data: Validated[pd.Series, Finite, NonEmpty],
    window: Hyper[int, Ge[2]] = 20,
    num_std: Hyper[float, Gt[0.0]] = 2.0,
) -> pd.DataFrame:
    """Calculate Bollinger Bands.

    Bollinger Bands consist of a middle band (SMA) and two outer bands that
    are standard deviations away from the middle band. They expand and contract
    based on market volatility.

    Components:
    - Middle Band = SMA(close, window)
    - Upper Band = Middle Band + (std_dev * num_std)
    - Lower Band = Middle Band - (std_dev * num_std)
    - Bandwidth = (Upper Band - Lower Band) / Middle Band
    - %B = (Price - Lower Band) / (Upper Band - Lower Band)

    Interpretation:
    - Price near upper band: Overbought
    - Price near lower band: Oversold
    - Bands squeezing: Low volatility, potential breakout coming
    - Bands expanding: High volatility, trend in motion
    - %B > 1: Price above upper band (very overbought)
    - %B < 0: Price below lower band (very oversold)
    - %B = 0.5: Price at middle band

    Common strategies:
    - Mean reversion: Sell at upper band, buy at lower band
    - Breakout: Buy when price breaks above upper band with expanding bands
    - Squeeze: Enter when bands squeeze then expand (volatility breakout)
    - Walking the bands: Strong trends "walk" along one band

    Features:
    - Pandas optimized (rolling window operations)
    - Configurable window and standard deviation multiplier
    - Returns all components (bands, bandwidth, %B)
    - Works with any numeric column

    Args:
      data: Input Series.
      window: Rolling window size (default: 20)
      num_std: Number of standard deviations for bands (default: 2.0)

    Returns:
      DataFrame with 'bb_middle', 'bb_upper', 'bb_lower', 'bb_bandwidth', 'bb_percent' columns

    Raises:
      ValueError: If data contains NaN/Inf

    Example:
      >>> import pandas as pd
      >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113, 115, 117, 116, 118, 120])
      >>> result = bollinger_bands(prices, window=10, num_std=2.0)
      >>> # Returns DataFrame with all Bollinger Band components
    """
    # Calculate middle band (SMA)
    middle = data.rolling(window=window, min_periods=1).mean()

    # Calculate standard deviation
    std = data.rolling(window=window, min_periods=1).std()

    # Calculate upper and lower bands
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)

    # Calculate bandwidth (volatility measure)
    # Avoid division by zero
    bandwidth = np.where(middle > 0, (upper - lower) / middle, np.nan)

    # Calculate %B (position within bands)
    # Avoid division by zero
    band_range = upper - lower
    percent_b = np.where(
        band_range > 0,
        (data - lower) / band_range,
        0.5,  # Default to middle if bands collapsed
    )

    # Create result dataframe
    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_bandwidth": bandwidth,
            "bb_percent": percent_b,
        },
        index=data.index,
    )

"""Technical indicators module.

This module provides optimized technical analysis indicators using Numba JIT compilation.
Type checking is limited for Numba-compiled functions since numba doesn't provide type stubs.
"""
# pyright: reportAttributeAccessIssue=false, reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

from hipr import Ge, Gt, Hyper, Le, configurable
import numpy as np
import pandas as pd
from pdval import (
    Finite,
    Validated,
    validated,
)

from indikator._legs_numba import compute_zigzag_legs_numba


@configurable
@validated
def zigzag_legs(
    data: Validated[pd.Series, Finite],
    threshold: Hyper[float, Ge[0.0], Le[1.0]] = 0.01,
    min_distance_pct: Hyper[float, Ge[0.0], Le[1.0]] = 0.005,
    confirmation_bars: Hyper[int, Ge[0]] = 2,
    epsilon: Hyper[float, Gt[0.0]] = 1e-9,
) -> pd.Series:
    """Calculate zigzag leg count with market structure tracking.

    Implements Elliott Wave-style leg counting that distinguishes between:
    - **Corrections**: Temporary moves against the trend (sign stays same)
    - **Trend Changes**: Structure breaks that reverse the sign

    The algorithm tracks STRUCTURE, not just price direction. A down move in a
    bullish structure remains positive until it breaks below the previous low.

    **Important**: The output tracks market STRUCTURE (bullish/bearish), not just
    current leg direction (up/down). Use this for Elliott Wave analysis or structure-based
    strategies. For simpler zigzag tracking, consider using trend direction instead.

    Features:
    - Signed output: positive for bullish structure, negative for bearish structure
    - Confirmation period to filter false reversals
    - Minimum distance filter to avoid counting tiny wicks
    - Numba-optimized for performance

    Args:
      data: Input Series (e.g., close prices)
      threshold: Minimum percentage change (0.01 = 1%) to trigger a reversal
      min_distance_pct: Minimum percentage move (0.005 = 0.5%) to update pivot
      confirmation_bars: Number of bars to confirm reversal (default 2)
      epsilon: Small value to prevent division by zero

    Returns:
      Series with zigzag leg counts

    Raises:
      ValueError: If data contains NaN or infinite values
      pandera.errors.SchemaError: If validation fails

    Example:
      >>> import pandas as pd
      >>> # Bullish structure: stays positive during corrections
      >>> data = pd.Series([100, 110, 105, 115])  # Up, down (correction), up
      >>> result = zigzag_legs(data, threshold=0.03, confirmation_bars=0)
      >>> # Output: [0, 1, 1, 1] - stays positive (bullish structure)
      >>>
      >>> # Structure break: negative when breaking previous low
      >>> data2 = pd.Series([100, 110, 105, 115, 100, 95])  # Breaks below 105
      >>> result2 = zigzag_legs(data2, threshold=0.03, confirmation_bars=0)
      >>> # After price breaks previous low at 105, structure changes to bearish
    """
    # We use valid data but need to ensure types for Numba
    closes = np.asarray(data.values, dtype=np.float64)

    legs = compute_zigzag_legs_numba(
        closes,
        threshold,
        min_distance_pct,
        confirmation_bars,
        epsilon,
    )

    return pd.Series(legs, index=data.index, name="zigzag_legs")

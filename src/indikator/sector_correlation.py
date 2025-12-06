"""Sector correlation indicator module.

This module calculates rolling correlation between a stock and its sector ETF
to measure how closely the stock moves with the broader sector.
"""
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from hipr import Ge, Hyper, Le  # noqa: TC002
import pandas as pd
from pdval import (  # noqa: TC002
    Finite,
    Validated,
    validated,
)

# Maximum ratio of NaN values allowed in aligned data before using default
MAX_NAN_RATIO = 0.5


@validated
def sector_correlation(
    stock_data: Validated[pd.Series, Finite],
    sector_data: Validated[pd.Series, Finite] | None = None,
    *,
    window: Hyper[int, Ge[2]] = 20,
    default_value: Hyper[float, Ge[-1.0], Le[1.0]] = 0.0,
) -> pd.Series:
    """Calculate rolling correlation between a stock and its sector/index.

    Measures how closely a stock moves with its sector or the broader market.
    - High correlation (> 0.8): Stock moves with the market
    - Low correlation (~ 0): Stock is moving independently
    - Negative correlation: Stock moves opposite to the market

    Args:
      stock_data: Stock price series (e.g., close prices)
      sector_data: Sector/Index price series. If None, returns default_value.
      window: Rolling window size for correlation calculation
      default_value: Value to return if sector_data is None or insufficient data

    Returns:
      Series with rolling correlation values named "sector_correlation"

    Raises:
      ValueError: If validation fails
    """
    if sector_data is None:
        return pd.Series(
            default_value, index=stock_data.index, name="sector_correlation"
        )

    # Align series on index
    aligned_stock, _ = stock_data.align(sector_data, join="inner")

    if len(aligned_stock) < window:
        return pd.Series(
            default_value, index=stock_data.index, name="sector_correlation"
        )

    # Align sector data to stock data index using forward fill
    # This handles cases where timestamps don't exactly match
    aligned_sector = sector_data.reindex(stock_data.index).ffill()

    # Check if alignment resulted in too many NaN values
    nan_ratio = aligned_sector.isna().sum() / len(aligned_sector)
    if nan_ratio > MAX_NAN_RATIO:
        # Poor alignment quality - return default value
        return pd.Series(
            default_value, index=stock_data.index, name="sector_correlation"
        )

    # Calculate rolling correlation
    correlation = stock_data.rolling(window=window).corr(aligned_sector)

    # Fill NaN values (start of window) with default value
    correlation = correlation.fillna(default_value)
    correlation.name = "sector_correlation"

    return correlation

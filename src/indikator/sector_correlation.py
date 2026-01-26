"""Sector Correlation indicator module.

This module calculates the rolling correlation between a stock and a sector/index.
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
from indikator.utils import to_numpy


@configurable
@validate
def sector_correlation(
  stock_data: Validated[pd.Series[float], Finite, NotEmpty],
  sector_data: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 20,
) -> IndicatorResult:
  """Calculate rolling correlation between a stock and its sector/index.

  Formula:
  Corr = RollingCorr(Stock, Sector, period)

  Interpretation:
  - High Corr (> 0.8): Moving with sector (Systematic risk dominates)
  - Low Corr (< 0.5): Independent movement (Idiosyncratic risk)
  - Negative Corr: Inverse movement (Hedge/Contra)

  Args:
    stock_data: Stock price Series.
    sector_data: Sector/Index price Series.
    period: Rolling correlation window (default: 20)

  Returns:
    IndicatorResult(index, correlation)
  """
  # Ensure alignment
  # Using pandas operations handles alignment automatically on index
  # This is efficient enough for correlation usually

  # Calculate rolling correlation
  # Note: Pandas aligns indices automatically before correlation
  corr_series = stock_data.rolling(window=period).corr(sector_data)

  # Extract array
  corr_arr = to_numpy(corr_series)

  # Handle potential NaNs not from window (e.g. misalignment) - kept as NaN

  return IndicatorResult(
    data_index=corr_series.index, value=corr_arr, name="correlation"
  )

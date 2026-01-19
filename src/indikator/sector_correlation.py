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
import numpy as np
import pandas as pd

from indikator._results import SectorCorrelationResult


@configurable
@validate
def sector_correlation(
  stock_data: Validated[pd.Series, Finite, NotEmpty],
  sector_data: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 20,
) -> SectorCorrelationResult:
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
    SectorCorrelationResult(index, correlation)
  """
  # Ensure alignment
  # Using pandas operations handles alignment automatically on index
  # This is efficient enough for correlation usually

  # Calculate rolling correlation
  # Note: Pandas aligns indices automatically before correlation
  corr_series = stock_data.rolling(window=period).corr(sector_data)

  # Extract array
  corr_arr = corr_series.to_numpy(dtype=np.float64, copy=False)

  # Handle potential NaNs not from window (e.g. misalignment) - kept as NaN

  return SectorCorrelationResult(index=corr_series.index, correlation=corr_arr)

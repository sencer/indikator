"""Churn Factor indicator module.

This module calculates Volume / (High - Low), which measures trading activity
relative to the price range. High churn indicates lots of trading within a
narrow price range.
"""

from typing import Literal

from hipr import Gt, Hyper, configurable
import numpy as np
import pandas as pd
from pdval import (
  Ge as ColsGe,
  HasColumns,
  NonNegative,
  Validated,
  validated,
)


@configurable
@validated
def churn_factor(
  data: Validated[
    pd.DataFrame,
    HasColumns[Literal["high", "low", "volume"]],
    ColsGe[Literal["high", "low"]],
    NonNegative,
  ],
  epsilon: Hyper[float, Gt[0.0]] = 1e-9,
  fill_strategy: str = "zero",
  fill_value: float | None = None,
) -> pd.DataFrame:
  """Calculate Churn Factor (Volume / High-Low Range).

  High churn indicates high volume with little price movement, suggesting
  indecision or potential reversal (accumulation/distribution).

  Args:
    data: OHLCV DataFrame
    epsilon: Small value to prevent division by zero
    fill_strategy: Strategy for handling zero range bars ('zero', 'nan', 'forward_fill')
    fill_value: Custom value to use when fill_strategy='zero' (default: 0.0)

  Returns:
    DataFrame with 'churn_factor' column added

  Raises:
    ValueError: If required columns missing
    pandera.errors.SchemaError: If validation fails
  """
  # Calculate range
  price_range = data["high"] - data["low"]

  # Initialize churn series based on strategy
  if fill_strategy in {"nan", "forward_fill"}:
    churn = pd.Series(np.nan, index=data.index)
  else:
    # Default to 0.0 or custom fill value
    initial_value = fill_value if fill_value is not None else 0.0
    churn = pd.Series(initial_value, index=data.index)

  # Calculate churn where range is significant
  valid_range = price_range > epsilon

  if valid_range.any():
    churn[valid_range] = data["volume"][valid_range] / price_range[valid_range]

  # Apply forward fill if requested
  if fill_strategy == "forward_fill":
    churn = churn.ffill()

  # Create result dataframe
  data_copy = data.copy()
  data_copy["churn_factor"] = churn

  return data_copy

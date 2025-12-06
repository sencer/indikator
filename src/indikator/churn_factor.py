"""Churn Factor indicator module.

This module calculates Volume / (High - Low), which measures trading activity
relative to the price range. High churn indicates lots of trading within a
narrow price range.
"""

from typing import TYPE_CHECKING, Literal, cast

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

if TYPE_CHECKING:
  from numpy.typing import NDArray


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
  price_range_values = cast("NDArray[np.float64]", price_range.values)
  volume_values = cast("NDArray[np.float64]", data["volume"].values)

  # Use vectorization with numpy where to handle division by zero
  # This is faster and cleaner than boolean indexing
  result_values = np.divide(
    volume_values,
    price_range_values,
    out=np.zeros_like(volume_values, dtype=np.float64),
    where=price_range_values > epsilon,
  )

  # Convert to Series
  churn = pd.Series(result_values, index=data.index)

  # Apply strategy for invalid values (where range <= epsilon)
  # The 'zero' strategy is already handled by the initialization of out=zeros_like above

  if fill_strategy == "nan":
    # Mask the zero values with NaN where range was invalid
    churn = churn.where(price_range > epsilon, np.nan)
  elif fill_strategy == "forward_fill":
    # First set invalid to NaN, then ffill
    churn = churn.where(price_range > epsilon, np.nan).ffill()

  # Handle custom fill value if provided and strategy is zero
  if fill_strategy == "zero" and fill_value is not None:
    churn = churn.where(price_range > epsilon, fill_value)

  # Create result dataframe
  data_copy = data.copy()
  data_copy["churn_factor"] = churn

  return data_copy

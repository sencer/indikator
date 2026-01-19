"""Churn Factor indicator module.

This module calculates the Churn Factor, which measures the relationship
between volume and price range.
"""

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Gt, Hyper, configurable
import numpy as np
import pandas as pd

from indikator._constants import DEFAULT_EPSILON
from indikator._results import ChurnFactorResult


@configurable
@validate
def churn_factor(
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  volume: Validated[pd.Series, Finite, NotEmpty],
  epsilon: Hyper[float, Gt[0.0]] = DEFAULT_EPSILON,
) -> ChurnFactorResult:
  """Calculate Churn Factor.

  Churn factor measures the efficiency of volume in moving the price.
  High churn means high volume but low price movement (indecision/turning point).

  Formula:
  Churn = Volume / (High - Low)

  Interpretation:
  - High Churn: High volume with tight range (distribution/accumulation)
  - Low Churn: Price moving freely on low volume (or low vol/low range)

  Args:
    high: High prices Series.
    low: Low prices Series.
    volume: Volume Series.
    epsilon: Division by zero protection.

  Returns:
    ChurnFactorResult(index, churn)
  """
  # Convert to numpy
  high_arr = high.to_numpy(dtype=np.float64, copy=False)
  low_arr = low.to_numpy(dtype=np.float64, copy=False)
  vol_arr = volume.to_numpy(dtype=np.float64, copy=False)

  price_range = high_arr - low_arr

  # Calculate Churn: Vol / Range
  # Handle zero range
  churn = np.zeros_like(vol_arr)

  valid_range = price_range > epsilon
  churn[valid_range] = vol_arr[valid_range] / price_range[valid_range]

  return ChurnFactorResult(index=high.index, churn=churn)

"""Accumulation/Distribution Line (AD) indicator module.

Measures buying and selling pressure through price-volume relationship.
"""

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import configurable
import pandas as pd

from indikator._ad_numba import compute_ad_numba
from indikator._results import ADResult
from indikator.utils import to_numpy


@configurable
@validate
def ad(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  volume: Validated[pd.Series[float], Finite, NotEmpty],
) -> ADResult:
  """Calculate Accumulation/Distribution Line (AD).

  The A/D Line measures cumulative buying/selling pressure by analyzing
  where the close falls within the high-low range, weighted by volume.

  Formula:
  CLV = ((Close - Low) - (High - Close)) / (High - Low)
  AD = cumsum(CLV * Volume)

  Interpretation:
  - Rising AD: Buying pressure (accumulation)
  - Falling AD: Selling pressure (distribution)
  - AD divergence from price: Potential reversal

  Features:
  - Single pass O(N) computation
  - No parameters required

  Args:
    high: High prices
    low: Low prices
    close: Close prices
    volume: Volume

  Returns:
    ADResult with cumulative A/D values

  Example:
    >>> result = ad(high, low, close, volume)
  """
  h = to_numpy(high)
  low_arr = to_numpy(low)
  c = to_numpy(close)
  v = to_numpy(volume)

  ad_values = compute_ad_numba(h, low_arr, c, v)

  return ADResult(data_index=close.index, ad=ad_values)

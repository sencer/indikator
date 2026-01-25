"""Accumulation/Distribution Line (AD) indicator module.

Measures buying and selling pressure through price-volume relationship.
"""

from typing import TYPE_CHECKING, cast

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import configurable
import numpy as np
import pandas as pd

if TYPE_CHECKING:
  from numpy.typing import NDArray

from indikator._ad_numba import compute_ad_numba
from indikator._results import ADResult


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
  h = cast(
    "NDArray[np.float64]",
    high.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )
  low_arr = cast(
    "NDArray[np.float64]",
    low.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )
  c = cast(
    "NDArray[np.float64]",
    close.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )
  v = cast(
    "NDArray[np.float64]",
    volume.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  ad_values = compute_ad_numba(h, low_arr, c, v)

  return ADResult(data_index=close.index, ad=ad_values)

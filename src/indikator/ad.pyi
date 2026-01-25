"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import Protocol, TypedDict

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import ADResult

class _ad_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    high: Validated[pd.Series[float], Finite, NotEmpty],
    low: Validated[pd.Series[float], Finite, NotEmpty],
    close: Validated[pd.Series[float], Finite, NotEmpty],
    volume: Validated[pd.Series[float], Finite, NotEmpty],
  ) -> ADResult: ...

class _ad_ConfigDict(TypedDict, total=False):
  pass

class _ad_Config(_NCMakeableModel[_ad_Bound]):
  """Configuration class for ad.

  Calculate Accumulation/Distribution Line (AD).

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

  pass

class ad:
  Type = _ad_Bound
  Config = _ad_Config
  ConfigDict = _ad_ConfigDict
  def __new__(
    cls,
    high: Validated[pd.Series[float], Finite, NotEmpty],
    low: Validated[pd.Series[float], Finite, NotEmpty],
    close: Validated[pd.Series[float], Finite, NotEmpty],
    volume: Validated[pd.Series[float], Finite, NotEmpty],
  ) -> ADResult: ...

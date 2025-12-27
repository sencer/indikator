"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import Protocol, TypedDict

from datawarden import Finite, NonEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

class _sector_correlation_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    stock_data: Validated[pd.Series, Finite, NonEmpty],
    sector_data: Validated[pd.Series, Finite, NonEmpty] | None = ...,
  ) -> pd.Series: ...

class _sector_correlation_ConfigDict(TypedDict, total=False):
  pass

class _sector_correlation_Config(_NCMakeableModel[_sector_correlation_Bound]):
  """Configuration class for sector_correlation.

  Calculate rolling correlation between a stock and its sector/index.

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

  pass

class sector_correlation:
  Type = _sector_correlation_Bound
  Config = _sector_correlation_Config
  ConfigDict = _sector_correlation_ConfigDict
  def __new__(
    cls,
    stock_data: Validated[pd.Series, Finite, NonEmpty],
    sector_data: Validated[pd.Series, Finite, NonEmpty] | None = ...,
  ) -> pd.Series: ...

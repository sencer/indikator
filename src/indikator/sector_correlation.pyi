"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import SectorCorrelationResult

class _sector_correlation_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self,
    stock_data: Validated[pd.Series, Finite, NotEmpty],
    sector_data: Validated[pd.Series, Finite, NotEmpty],
  ) -> SectorCorrelationResult: ...

class _sector_correlation_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for sector_correlation.

  Configuration:
      period (int)
  """

  period: int

class _sector_correlation_Config(_NCMakeableModel[_sector_correlation_Bound]):
  """Configuration class for sector_correlation.

  Calculate rolling correlation between a stock and its sector/index.

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

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for sector_correlation.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _sector_correlation_Bound: ...

class sector_correlation:
  Type = _sector_correlation_Bound
  Config = _sector_correlation_Config
  ConfigDict = _sector_correlation_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls,
    stock_data: Validated[pd.Series, Finite, NotEmpty],
    sector_data: Validated[pd.Series, Finite, NotEmpty],
    period: int = ...,
  ) -> SectorCorrelationResult: ...

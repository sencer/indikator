"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Literal, Protocol, TypedDict, override

from datawarden import Datetime, Finite, Index, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import PivotPointsResult

class _pivots_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def method(self) -> Literal["standard", "fibonacci", "woodie", "camarilla"]: ...
  @property
  def anchor(self) -> str: ...
  def __call__(
    self,
    high: Validated[pd.Series[float], Finite, Index(Datetime), NotEmpty],
    low: Validated[pd.Series[float], Finite, Index(Datetime), NotEmpty],
    close: Validated[pd.Series[float], Finite, Index(Datetime), NotEmpty],
  ) -> PivotPointsResult: ...

class _pivots_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for pivots.

  Configuration:
      method (Literal['standard', 'fibonacci', 'woodie', 'camarilla'])
      anchor (str)
  """

  method: Literal["standard", "fibonacci", "woodie", "camarilla"]
  anchor: str

class _pivots_Config(_NCMakeableModel[_pivots_Bound]):
  """Configuration class for pivots.

  Calculate Pivot Points.

  Pivot points are significant support and resistance levels derived from
  prior period's price action.

  Methods:
  - Standard: Floor pivots (Classic)
  - Fibonacci: Standard Pivot + Fibonacci extensions
  - Woodie: Weighted close, forward-looking
  - Camarilla: Mean reversion/Breakout levels

  Levels typically include:
  - P: Pivot Point (Central)
  - R1, R2, R3, R4: Resistance levels
  - S1, S2, S3, S4: Support levels

  Args:
    high: High prices with DatetimeIndex
    low: Low prices with DatetimeIndex
    close: Close prices with DatetimeIndex
    method: Calculation method (default: 'standard')
    anchor: Period to aggregate prior data (default: 'D' for Daily)

  Returns:
    PivotPointsResult(index, levels: dict)

  Configuration:
      method (Literal['standard', 'fibonacci', 'woodie', 'camarilla'])
      anchor (str)
  """

  method: Literal["standard", "fibonacci", "woodie", "camarilla"]
  anchor: str
  def __init__(
    self,
    *,
    method: Literal["standard", "fibonacci", "woodie", "camarilla"] = ...,
    anchor: str = ...,
  ) -> None: ...
  """Initialize configuration for pivots.

    Configuration:
        method (Literal['standard', 'fibonacci', 'woodie', 'camarilla'])
        anchor (str)
    """

  @override
  def make(self) -> _pivots_Bound: ...

class pivots:
  Type = _pivots_Bound
  Config = _pivots_Config
  ConfigDict = _pivots_ConfigDict
  method: ClassVar[Literal["standard", "fibonacci", "woodie", "camarilla"]]
  anchor: ClassVar[str]
  def __new__(
    cls,
    high: Validated[pd.Series[float], Finite, Index(Datetime), NotEmpty],
    low: Validated[pd.Series[float], Finite, Index(Datetime), NotEmpty],
    close: Validated[pd.Series[float], Finite, Index(Datetime), NotEmpty],
    method: Literal["standard", "fibonacci", "woodie", "camarilla"] = ...,
    anchor: str = ...,
  ) -> PivotPointsResult: ...

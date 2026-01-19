"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import SARResult

class _sar_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def acceleration(self) -> float: ...
  @property
  def maximum(self) -> float: ...
  def __call__(
    self,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
  ) -> SARResult: ...

class _sar_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for sar.

  Configuration:
      acceleration (float)
      maximum (float)
  """

  acceleration: float
  maximum: float

class _sar_Config(_NCMakeableModel[_sar_Bound]):
  """Configuration class for sar.

  Calculate Parabolic SAR (Stop and Reverse).

  Parabolic SAR provides potential entry and exit points by trailing
  price with an accelerating stop level.

  Formula:
  SAR(t+1) = SAR(t) + AF * (EP - SAR(t))

  Where:
  - AF: Acceleration Factor (starts at 0.02, increments by 0.02 on new EP)
  - EP: Extreme Point (highest high in uptrend, lowest low in downtrend)

  Interpretation:
  - Price above SAR: Uptrend (SAR is support)
  - Price below SAR: Downtrend (SAR is resistance)
  - SAR flip: Potential trend reversal

  Features:
  - State machine with register optimization
  - Handles trend reversals automatically

  Args:
    high: High prices
    low: Low prices
    acceleration: AF start and increment (default: 0.02)
    maximum: Maximum AF (default: 0.2)

  Returns:
    SARResult with SAR values

  Example:
    >>> result = sar(high, low, acceleration=0.02, maximum=0.2)

  Configuration:
      acceleration (float)
      maximum (float)
  """

  acceleration: float
  maximum: float
  def __init__(self, *, acceleration: float = ..., maximum: float = ...) -> None: ...
  """Initialize configuration for sar.

    Configuration:
        acceleration (float)
        maximum (float)
    """

  @override
  def make(self) -> _sar_Bound: ...

class sar:
  Type = _sar_Bound
  Config = _sar_Config
  ConfigDict = _sar_ConfigDict
  acceleration: ClassVar[float]
  maximum: ClassVar[float]
  def __new__(
    cls,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    acceleration: float = ...,
    maximum: float = ...,
  ) -> SARResult: ...

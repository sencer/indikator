"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

class _mavp_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def minperiod(self) -> int: ...
  @property
  def maxperiod(self) -> int: ...
  def __call__(
    self,
    data: Validated[pd.Series[float], NotEmpty],
    periods: Validated[pd.Series[float], NotEmpty],
    matype: int = ...,
  ) -> pd.Series: ...

class _mavp_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for mavp.

  Configuration:
      minperiod (int)
      maxperiod (int)
  """

  minperiod: int
  maxperiod: int

class _mavp_Config(_NCMakeableModel[_mavp_Bound]):
  """Configuration class for mavp.

  Moving Average with Variable Period.

  Calculates a moving average where the period varies per element.
  Currently supports SMA (Simple Moving Average) logic (matype=0).

  Args:
    data: Input price series.
    periods: Series containing the period to use for each element.
             Values are clamped between minperiod and maxperiod.
    minperiod: Minimum period allowed (default: 2).
    maxperiod: Maximum period allowed (default: 30).
    matype: Moving Average Type (0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA).

  Returns:
    pd.Series: MAVP values.

  Configuration:
      minperiod (int)
      maxperiod (int)
  """

  minperiod: int
  maxperiod: int
  def __init__(self, *, minperiod: int = ..., maxperiod: int = ...) -> None: ...
  """Initialize configuration for mavp.

    Configuration:
        minperiod (int)
        maxperiod (int)
    """

  @override
  def make(self) -> _mavp_Bound: ...

class mavp:
  Type = _mavp_Bound
  Config = _mavp_Config
  ConfigDict = _mavp_ConfigDict
  minperiod: ClassVar[int]
  maxperiod: ClassVar[int]
  def __new__(
    cls,
    data: Validated[pd.Series[float], NotEmpty],
    periods: Validated[pd.Series[float], NotEmpty],
    matype: int = ...,
    minperiod: int = ...,
    maxperiod: int = ...,
  ) -> pd.Series: ...

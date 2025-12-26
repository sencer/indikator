"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd
from validated import Finite, NonEmpty, Validated

class _slope_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def window(self) -> int: ...
  def __call__(self, data: Validated[pd.Series, Finite, NonEmpty]) -> pd.Series: ...

class _slope_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for slope.

  Configuration:
      window (int)
  """

  window: int

class _slope_Config(_NCMakeableModel[_slope_Bound]):
  """Configuration class for slope.

  Calculate the slope of linear regression over a rolling window.

  The slope indicates the direction and steepness of the trend:
  - Positive slope: Uptrend
  - Negative slope: Downtrend
  - Near zero: Sideways/Consolidation

  Args:
    data: Series of prices (e.g., close prices)
    window: Rolling window size for regression

  Returns:
    Series with slope values

  Raises:
    ValueError: If validation fails

  Configuration:
      window (int)
  """

  window: int
  def __init__(self, *, window: int = ...) -> None: ...
  """Initialize configuration for slope.

    Configuration:
        window (int)
    """

  @override
  def make(self) -> _slope_Bound: ...

class slope:
  Type = _slope_Bound
  Config = _slope_Config
  ConfigDict = _slope_ConfigDict
  window: ClassVar[int]
  def __new__(
    cls, data: Validated[pd.Series, Finite, NonEmpty], window: int = ...
  ) -> pd.Series: ...

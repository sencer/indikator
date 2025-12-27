"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Literal, Protocol, TypedDict, override

from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd
from validated import Ge as ColsGe, HasColumns, NonEmpty, NonNegative, Validated

class _churn_factor_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def epsilon(self) -> float: ...
  def __call__(
    self,
    data: Validated[
      pd.DataFrame,
      HasColumns(["high", "low", "volume"]),
      ColsGe("high", "low"),
      NonNegative,
      NonEmpty,
    ],
    fill_strategy: Literal["zero", "nan", "forward_fill"] = ...,
    fill_value: float | None = ...,
  ) -> pd.Series: ...

class _churn_factor_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for churn_factor.

  Configuration:
      epsilon (float)
  """

  epsilon: float

class _churn_factor_Config(_NCMakeableModel[_churn_factor_Bound]):
  """Configuration class for churn_factor.

  Calculate Churn Factor (Volume / High-Low Range).

  High churn indicates high volume with little price movement, suggesting
  indecision or potential reversal (accumulation/distribution).

  Args:
    data: OHLCV DataFrame
    epsilon: Small value to prevent division by zero
    fill_strategy: Strategy for handling zero range bars ('zero', 'nan', 'forward_fill')
    fill_value: Custom value to use when fill_strategy='zero' (default: 0.0)

  Returns:
    DataFrame with 'churn_factor' column

  Raises:
    ValueError: If required columns missing or validation fails

  Configuration:
      epsilon (float)
  """

  epsilon: float
  def __init__(self, *, epsilon: float = ...) -> None: ...
  """Initialize configuration for churn_factor.

    Configuration:
        epsilon (float)
    """

  @override
  def make(self) -> _churn_factor_Bound: ...

class churn_factor:
  Type = _churn_factor_Bound
  Config = _churn_factor_Config
  ConfigDict = _churn_factor_ConfigDict
  epsilon: ClassVar[float]
  def __new__(
    cls,
    data: Validated[
      pd.DataFrame,
      HasColumns(["high", "low", "volume"]),
      ColsGe("high", "low"),
      NonNegative,
      NonEmpty,
    ],
    fill_strategy: Literal["zero", "nan", "forward_fill"] = ...,
    fill_value: float | None = ...,
    epsilon: float = ...,
  ) -> pd.Series: ...

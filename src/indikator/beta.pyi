"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import BETAResult

class _beta_statistical_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self,
    x: Validated[pd.Series[float], Finite, NotEmpty],
    y: Validated[pd.Series[float], Finite, NotEmpty],
  ) -> BETAResult: ...

class _beta_statistical_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for beta_statistical.

  Configuration:
      period (int)
  """

  period: int

class _beta_statistical_Config(_NCMakeableModel[_beta_statistical_Bound]):
  """Configuration class for beta_statistical.

  Calculate rolling BETA coefficient on RAW INPUTS.

  This is the pure statistical calculation of Beta:
    BETA = cov(X, Y) / var(X)

  This function does NOT transform inputs into returns. It calculates
  beta directly on the provided x and y series.

  Use this if you have already calculated returns or want beta of prices.

  Args:
    x: Independent variable (e.g., market returns)
    y: Dependent variable (e.g., stock returns)
    period: Rolling window size (default: 5)

  Returns:
    BETAResult(index, beta)

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for beta_statistical.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _beta_statistical_Bound: ...

class beta_statistical:
  Type = _beta_statistical_Bound
  Config = _beta_statistical_Config
  ConfigDict = _beta_statistical_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls,
    x: Validated[pd.Series[float], Finite, NotEmpty],
    y: Validated[pd.Series[float], Finite, NotEmpty],
    period: int = ...,
  ) -> BETAResult: ...

class _beta_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self,
    x: Validated[pd.Series[float], Finite, NotEmpty],
    y: Validated[pd.Series[float], Finite, NotEmpty],
  ) -> BETAResult: ...

class _beta_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for beta.

  Configuration:
      period (int)
  """

  period: int

class _beta_Config(_NCMakeableModel[_beta_Bound]):
  """Configuration class for beta.

  Calculate rolling BETA coefficient (TA-Lib compatible).

  Matches TA-Lib's behavior: automatically calculates 1-period
  percentage change (returns) for both inputs before calculating Beta.

  BETA = cov(rocp(x), rocp(y)) / var(rocp(x))

  Performance Note:
  Uses a FUSED kernel that calculates ROCP on the fly within the rolling loop,
  avoiding intermediate array allocations.

  Args:
    x: Independent variable price series (e.g., market index)
    y: Dependent variable price series (e.g., stock price)
    period: Rolling window size (default: 5)

  Returns:
    BETAResult(index, beta)

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for beta.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _beta_Bound: ...

class beta:
  Type = _beta_Bound
  Config = _beta_Config
  ConfigDict = _beta_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls,
    x: Validated[pd.Series[float], Finite, NotEmpty],
    y: Validated[pd.Series[float], Finite, NotEmpty],
    period: int = ...,
  ) -> BETAResult: ...

"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import DEMAResult

class _dema_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(self, data: Validated[pd.Series, Finite, NotEmpty]) -> DEMAResult: ...

class _dema_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for dema.

  Configuration:
      period (int)
  """

  period: int

class _dema_Config(_NCMakeableModel[_dema_Bound]):
  """Configuration class for dema.

  Calculate Double Exponential Moving Average (DEMA).

  DEMA reduces the lag inherent in EMAs by applying a correction factor
  based on the difference between EMA and EMA of EMA.

  Formula:
  DEMA = 2 * EMA(P, n) - EMA(EMA(P, n), n)

  Interpretation:
  - Faster response to price changes than standard EMA
  - Less whipsaw in trending markets
  - Better for crossover strategies

  Features:
  - Fused Numba kernel: computes both EMA stages in single loop
  - Values stay in registers, no intermediate arrays

  Args:
    data: Input price Series (typically close prices)
    period: Lookback period (default: 20)

  Returns:
    DEMAResult with DEMA values

  Example:
    >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108])
    >>> result = dema(prices, period=5)

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for dema.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _dema_Bound: ...

class dema:
  Type = _dema_Bound
  Config = _dema_Config
  ConfigDict = _dema_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls, data: Validated[pd.Series, Finite, NotEmpty], period: int = ...
  ) -> DEMAResult: ...

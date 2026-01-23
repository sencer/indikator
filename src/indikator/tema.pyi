"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar
from typing import Protocol
from typing import TypedDict
from typing import override

from nonfig import MakeableModel as _NCMakeableModel

from datawarden import Finite, NotEmpty, Validated
import pandas as pd
from indikator._results import TEMAResult

class _tema_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(self, data: Validated[pd.Series, Finite, NotEmpty]) -> TEMAResult: ...

class _tema_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for tema.

  Configuration:
      period (int)
  """

  period: int

class _tema_Config(_NCMakeableModel[_tema_Bound]):
  """Configuration class for tema.

  Calculate Triple Exponential Moving Average (TEMA).

  TEMA minimizes lag using a combination of single, double, and triple
  smoothed EMAs with specific weights.

  Formula:
  TEMA = 3 * EMA1 - 3 * EMA2 + EMA3

  Where:
  - EMA1 = EMA(price, n)
  - EMA2 = EMA(EMA1, n)
  - EMA3 = EMA(EMA2, n)

  Interpretation:
  - Even faster response than DEMA
  - Best for capturing short-term trend changes
  - Can be combined with slower MAs for crossover strategies

  Features:
  - Fused Numba kernel: computes all three EMA stages in single loop
  - Values stay in registers, no intermediate arrays

  Args:
    data: Input price Series (typically close prices)
    period: Lookback period (default: 20)

  Returns:
    TEMAResult with TEMA values

  Example:
    >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108])
    >>> result = tema(prices, period=5)

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for tema.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _tema_Bound: ...

class tema:
  Type = _tema_Bound
  Config = _tema_Config
  ConfigDict = _tema_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls, data: Validated[pd.Series, Finite, NotEmpty], period: int = ...
  ) -> TEMAResult: ...

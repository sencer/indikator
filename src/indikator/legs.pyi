"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Literal, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import ZigzagLegsResult

class _legs_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def deviation(self) -> float: ...
  def __call__(
    self,
    high: Validated[pd.Series[float], Finite, NotEmpty],
    low: Validated[pd.Series[float], Finite, NotEmpty],
    close: Validated[pd.Series[float], Finite, NotEmpty],
    method: Literal["percentage", "absolute"] = ...,
  ) -> ZigzagLegsResult: ...

class _legs_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for legs.

  Configuration:
      deviation (float)
  """

  deviation: float

class _legs_Config(_NCMakeableModel[_legs_Bound]):
  """Configuration class for legs.

  Calculate ZigZag legs.

  Identifies swing highs and lows that exceed a minimum deviation threshold.
  Used to filter out noise and identify significant market moves.

  Features:
  - Numba-optimized
  - "Percentage" or "Absolute" deviation modes
  - Returns structured legs data (start/end price, direction)

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    deviation: Minimum change to qualify as a new leg (0.05 = 5%).
    method: 'percentage' or 'absolute'.

  Returns:

  Raises:
    ValueError: If data contains NaN or infinite values

  Example:
    >>> import pandas as pd
    >>> # Bullish structure: stays positive during corrections
    >>> data = pd.Series([100, 110, 105, 115])  # Up, down (correction), up
    >>> result = zigzag_legs(data, threshold=0.03, confirmation_bars=0)
    >>> # Output: [0, 1, 1, 1] - stays positive (bullish structure)
    >>>
    >>> # Structure break: negative when breaking previous low
    >>> data2 = pd.Series([100, 110, 105, 115, 100, 95])  # Breaks below 105
    >>> result2 = zigzag_legs(data2, threshold=0.03, confirmation_bars=0)
    >>> # After price breaks previous low at 105, structure changes to bearish

  Configuration:
      deviation (float)
  """

  deviation: float
  def __init__(self, *, deviation: float = ...) -> None: ...
  """Initialize configuration for legs.

    Configuration:
        deviation (float)
    """

  @override
  def make(self) -> _legs_Bound: ...

class legs:
  Type = _legs_Bound
  Config = _legs_Config
  ConfigDict = _legs_ConfigDict
  deviation: ClassVar[float]
  def __new__(
    cls,
    high: Validated[pd.Series[float], Finite, NotEmpty],
    low: Validated[pd.Series[float], Finite, NotEmpty],
    close: Validated[pd.Series[float], Finite, NotEmpty],
    method: Literal["percentage", "absolute"] = ...,
    deviation: float = ...,
  ) -> ZigzagLegsResult: ...

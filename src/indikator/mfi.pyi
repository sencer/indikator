"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, Ge as GeValidator, HasColumns, NonEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

class _mfi_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def window(self) -> int: ...
  @property
  def epsilon(self) -> float: ...
  def __call__(
    self,
    data: Validated[
      pd.DataFrame,
      HasColumns(["high", "low", "close", "volume"]),
      GeValidator("high", "low"),
      Finite,
      NonEmpty,
    ],
  ) -> pd.Series: ...

class _mfi_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for mfi.

  Configuration:
      window (int)
      epsilon (float)
  """

  window: int
  epsilon: float

class _mfi_Config(_NCMakeableModel[_mfi_Bound]):
  """Configuration class for mfi.

  Calculate Money Flow Index (MFI).

  MFI is a momentum indicator that uses both price and volume to measure
  buying and selling pressure. It is also known as volume-weighted RSI.

  Formula:
  1. Typical Price = (High + Low + Close) / 3
  2. Money Flow = Typical Price * Volume
  3. Positive Money Flow = sum of MF when typical price increases
  4. Negative Money Flow = sum of MF when typical price decreases
  5. Money Flow Ratio = Positive Money Flow / Negative Money Flow
  6. MFI = 100 - (100 / (1 + Money Flow Ratio))

  Interpretation:
  - MFI > 80: Overbought (potential reversal down)
  - MFI < 20: Oversold (potential reversal up)
  - Divergence: MFI moves opposite to price (strong reversal signal)
  - Failure Swings: MFI crosses above 80 then below (sell) or below 20 then above (buy)

  Features:
  - Numba-optimized for performance
  - Uses typical price (H+L+C)/3
  - Handles division by zero with epsilon
  - 0-100 bounded range

  Args:
    data: OHLCV DataFrame
    window: Rolling window size (default: 14)
    epsilon: Small value to prevent division by zero

  Returns:
    Series with MFI values (0-100 range, NaN for initial bars)

  Raises:
    ValueError: If required columns missing or data contains NaN/Inf

  Example:
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'high': [10, 12, 11, 13, 15],
    ...     'low': [9, 10, 9, 11, 12],
    ...     'close': [9.5, 11, 10, 12, 14],
    ...     'volume': [100, 150, 120, 200, 180]
    ... })
    >>> result = mfi(data, window=3)
    >>> # Returns DataFrame with 'mfi' column

  Configuration:
      window (int)
      epsilon (float)
  """

  window: int
  epsilon: float
  def __init__(self, *, window: int = ..., epsilon: float = ...) -> None: ...
  """Initialize configuration for mfi.

    Configuration:
        window (int)
        epsilon (float)
    """

  @override
  def make(self) -> _mfi_Bound: ...

class mfi:
  Type = _mfi_Bound
  Config = _mfi_Config
  ConfigDict = _mfi_ConfigDict
  window: ClassVar[int]
  epsilon: ClassVar[float]
  def __new__(
    cls,
    data: Validated[
      pd.DataFrame,
      HasColumns(["high", "low", "close", "volume"]),
      GeValidator("high", "low"),
      Finite,
      NonEmpty,
    ],
    window: int = ...,
    epsilon: float = ...,
  ) -> pd.Series: ...

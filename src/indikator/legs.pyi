"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd
from validated import Finite, NonEmpty, NonNaN, Validated

class _zigzag_legs_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def threshold(self) -> float: ...
  @property
  def min_distance_pct(self) -> float: ...
  @property
  def confirmation_bars(self) -> int: ...
  @property
  def epsilon(self) -> float: ...
  def __call__(
    self, data: Validated[pd.Series, Finite, NonEmpty, NonNaN]
  ) -> pd.Series: ...

class _zigzag_legs_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for zigzag_legs.

  Configuration:
      threshold (float)
      min_distance_pct (float)
      confirmation_bars (int)
      epsilon (float)
  """

  threshold: float
  min_distance_pct: float
  confirmation_bars: int
  epsilon: float

class _zigzag_legs_Config(_NCMakeableModel[_zigzag_legs_Bound]):
  """Configuration class for zigzag_legs.

  Calculate zigzag leg count with market structure tracking.

  Implements Elliott Wave-style leg counting that distinguishes between:
  - **Corrections**: Temporary moves against the trend (sign stays same)
  - **Trend Changes**: Structure breaks that reverse the sign

  The algorithm tracks STRUCTURE, not just price direction. A down move in a
  bullish structure remains positive until it breaks below the previous low.

  **Important**: The output tracks market STRUCTURE (bullish/bearish), not just
  current leg direction (up/down). Use this for Elliott Wave analysis or structure-based
  strategies. For simpler zigzag tracking, consider using trend direction instead.

  Features:
  - Signed output: positive for bullish structure, negative for bearish structure
  - Confirmation period to filter false reversals
  - Minimum distance filter to avoid counting tiny wicks
  - Numba-optimized for performance

  Args:
    data: Input Series (e.g., close prices)
    threshold: Minimum percentage change (0.01 = 1%) to trigger a reversal
    min_distance_pct: Minimum percentage move (0.005 = 0.5%) to update pivot
    confirmation_bars: Number of bars to confirm reversal (default 2)
    epsilon: Small value to prevent division by zero

  Returns:
    Series with zigzag leg counts

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
      threshold (float)
      min_distance_pct (float)
      confirmation_bars (int)
      epsilon (float)
  """

  threshold: float
  min_distance_pct: float
  confirmation_bars: int
  epsilon: float
  def __init__(
    self,
    *,
    threshold: float = ...,
    min_distance_pct: float = ...,
    confirmation_bars: int = ...,
    epsilon: float = ...,
  ) -> None: ...
  """Initialize configuration for zigzag_legs.

    Configuration:
        threshold (float)
        min_distance_pct (float)
        confirmation_bars (int)
        epsilon (float)
    """

  @override
  def make(self) -> _zigzag_legs_Bound: ...

class zigzag_legs:
  Type = _zigzag_legs_Bound
  Config = _zigzag_legs_Config
  ConfigDict = _zigzag_legs_ConfigDict
  threshold: ClassVar[float]
  min_distance_pct: ClassVar[float]
  confirmation_bars: ClassVar[int]
  epsilon: ClassVar[float]
  def __new__(
    cls,
    data: Validated[pd.Series, Finite, NonEmpty, NonNaN],
    threshold: float = ...,
    min_distance_pct: float = ...,
    confirmation_bars: int = ...,
    epsilon: float = ...,
  ) -> pd.Series: ...

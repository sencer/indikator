"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import Protocol, TypedDict

from datawarden import Finite, HasColumns, NonEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

class _obv_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    data: Validated[pd.DataFrame, HasColumns(["close", "volume"]), Finite, NonEmpty],
  ) -> pd.Series: ...

class _obv_ConfigDict(TypedDict, total=False):
  pass

class _obv_Config(_NCMakeableModel[_obv_Bound]):
  """Configuration class for obv.

  Calculate On-Balance Volume (OBV).

  OBV is a cumulative indicator that adds volume on up days and subtracts
  volume on down days. It measures buying and selling pressure.

  Formula:
  - If close > previous close: OBV = OBV_previous + volume
  - If close < previous close: OBV = OBV_previous - volume
  - If close == previous close: OBV = OBV_previous

  Theory:
  - Volume precedes price (smart money accumulates before price rises)
  - Rising OBV with rising price = confirmed uptrend
  - Falling OBV with falling price = confirmed downtrend
  - OBV rising while price flat = accumulation (bullish)
  - OBV falling while price flat = distribution (bearish)

  Interpretation:
  - OBV trending up: Buying pressure increasing
  - OBV trending down: Selling pressure increasing
  - OBV divergence from price: Warning of potential reversal
  - OBV breakout before price: Early signal of price breakout

  Common strategies:
  - Trend confirmation: OBV should move with price trend
  - Divergence: OBV makes higher low while price makes lower low = bullish
  - Breakout confirmation: OBV breaks out with price = strong signal
  - Volume accumulation: Rising OBV during consolidation = breakout coming

  Features:
  - Numba-optimized for performance
  - Cumulative calculation (no window parameter)
  - Handles flat price days (no volume change)
  - Simple and effective

  Args:
    data: OHLCV DataFrame with 'close' and 'volume' columns

  Returns:
    DataFrame with 'obv' column

  Raises:
    ValueError: If required columns missing or data contains NaN/Inf

  Example:
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'close': [100, 102, 101, 103, 105],
    ...     'volume': [1000, 1200, 900, 1500, 1100]
    ... })
    >>> result = obv(data)
    >>> # OBV = [1000, 2200, 1300, 2800, 3900]
  """

  pass

class obv:
  Type = _obv_Bound
  Config = _obv_Config
  ConfigDict = _obv_ConfigDict
  def __new__(
    cls,
    data: Validated[pd.DataFrame, HasColumns(["close", "volume"]), Finite, NonEmpty],
  ) -> pd.Series: ...

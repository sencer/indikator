"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import Protocol, TypedDict

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import OBVResult

class _obv_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    close: Validated[pd.Series, Finite, NotEmpty],
    volume: Validated[pd.Series, Finite, NotEmpty],
  ) -> OBVResult: ...

class _obv_ConfigDict(TypedDict, total=False):
  pass

class _obv_Config(_NCMakeableModel[_obv_Bound]):
  """Configuration class for obv.

  Calculate On Balance Volume (OBV).

  OBV measures buying and selling pressure as a cumulative indicator that
  adds volume on up days and subtracts volume on down days.

  Formula:
  If Close > Close_prev: OBV = OBV_prev + Volume
  If Close < Close_prev: OBV = OBV_prev - Volume
  If Close = Close_prev: OBV = OBV_prev

  Interpretation:
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
    close: Validated[pd.Series, Finite, NotEmpty],
    volume: Validated[pd.Series, Finite, NotEmpty],
  ) -> OBVResult: ...

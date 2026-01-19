"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import ADXResult, ADXSingleResult

class _adx_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> ADXSingleResult: ...

class _adx_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for adx.

  Configuration:
      period (int)
  """

  period: int

class _adx_Config(_NCMakeableModel[_adx_Bound]):
  """Configuration class for adx.

  Calculate Average Directional Index (ADX).

  ADX measures trend strength regardless of direction. This function returns
  only the ADX series for maximum performance (matching TA-Lib).

  For Directional Indicators (+DI, -DI), use `adx_with_di()`.

  Interpretation:
  - ADX < 20: Weak trend / ranging market
  - ADX 25-50: Strong trend
  - ADX > 75: Extremely strong trend

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    ADXSingleResult(index, adx)

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for adx.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _adx_Bound: ...

class adx:
  Type = _adx_Bound
  Config = _adx_Config
  ConfigDict = _adx_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
    period: int = ...,
  ) -> ADXSingleResult: ...

class _adx_with_di_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> ADXResult: ...

class _adx_with_di_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for adx_with_di.

  Configuration:
      period (int)
  """

  period: int

class _adx_with_di_Config(_NCMakeableModel[_adx_with_di_Bound]):
  """Configuration class for adx_with_di.

  Calculate Average Directional Index (ADX) with DI components.

  Extended calculation that returns +DI and -DI alongside ADX.

  Components:
  - ADX: Average Directional Index (trend strength)
  - +DI: Plus Directional Indicator (bullish pressure)
  - -DI: Minus Directional Indicator (bearish pressure)

  Directional Indicators:
  - +DI > -DI: Bullish
  - -DI > +DI: Bearish
  - +DI crossing above -DI: Buy signal
  - -DI crossing above +DI: Sell signal

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    ADXResult object with adx, plus_di, minus_di series.

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for adx_with_di.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _adx_with_di_Bound: ...

class adx_with_di:
  Type = _adx_with_di_Bound
  Config = _adx_with_di_Config
  ConfigDict = _adx_with_di_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
    period: int = ...,
  ) -> ADXResult: ...

class _plus_dm_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _plus_dm_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for plus_dm.

  Configuration:
      period (int)
  """

  period: int

class _plus_dm_Config(_NCMakeableModel[_plus_dm_Bound]):
  """Configuration class for plus_dm.

  Calculate Plus Directional Movement (+DM).

  Returns the smoothed accumulated +DM over the period.

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    Series with +DM values.

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for plus_dm.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _plus_dm_Bound: ...

class plus_dm:
  Type = _plus_dm_Bound
  Config = _plus_dm_Config
  ConfigDict = _plus_dm_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
    period: int = ...,
  ) -> pd.Series: ...

class _minus_dm_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _minus_dm_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for minus_dm.

  Configuration:
      period (int)
  """

  period: int

class _minus_dm_Config(_NCMakeableModel[_minus_dm_Bound]):
  """Configuration class for minus_dm.

  Calculate Minus Directional Movement (-DM).

  Returns the smoothed accumulated -DM over the period.

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    Series with -DM values.

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for minus_dm.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _minus_dm_Bound: ...

class minus_dm:
  Type = _minus_dm_Bound
  Config = _minus_dm_Config
  ConfigDict = _minus_dm_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
    period: int = ...,
  ) -> pd.Series: ...

class _plus_di_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _plus_di_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for plus_di.

  Configuration:
      period (int)
  """

  period: int

class _plus_di_Config(_NCMakeableModel[_plus_di_Bound]):
  """Configuration class for plus_di.

  Calculate Plus Directional Indicator (+DI).

  +DI = 100 * (+DM / TR) (Smoothed)

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    Series with +DI values.

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for plus_di.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _plus_di_Bound: ...

class plus_di:
  Type = _plus_di_Bound
  Config = _plus_di_Config
  ConfigDict = _plus_di_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
    period: int = ...,
  ) -> pd.Series: ...

class _minus_di_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _minus_di_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for minus_di.

  Configuration:
      period (int)
  """

  period: int

class _minus_di_Config(_NCMakeableModel[_minus_di_Bound]):
  """Configuration class for minus_di.

  Calculate Minus Directional Indicator (-DI).

  -DI = 100 * (-DM / TR) (Smoothed)

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    Series with -DI values.

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for minus_di.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _minus_di_Bound: ...

class minus_di:
  Type = _minus_di_Bound
  Config = _minus_di_Config
  ConfigDict = _minus_di_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
    period: int = ...,
  ) -> pd.Series: ...

class _dx_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _dx_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for dx.

  Configuration:
      period (int)
  """

  period: int

class _dx_Config(_NCMakeableModel[_dx_Bound]):
  """Configuration class for dx.

  Calculate Directional Movement Index (DX).

  DX = 100 * |+DI - -DI| / (+DI + -DI)

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    Series with DX values.

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for dx.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _dx_Bound: ...

class dx:
  Type = _dx_Bound
  Config = _dx_Config
  ConfigDict = _dx_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
    period: int = ...,
  ) -> pd.Series: ...

class _adxr_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _adxr_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for adxr.

  Configuration:
      period (int)
  """

  period: int

class _adxr_Config(_NCMakeableModel[_adxr_Bound]):
  """Configuration class for adxr.

  Calculate Average Directional Movement Rating (ADXR).

  ADXR = (ADX + ADX[i - period]) / 2

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    Series with ADXR values.

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for adxr.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _adxr_Bound: ...

class adxr:
  Type = _adxr_Bound
  Config = _adxr_Config
  ConfigDict = _adxr_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
    period: int = ...,
  ) -> pd.Series: ...

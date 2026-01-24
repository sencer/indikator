"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import CCIResult

class _cci_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
    constant: float = ...,
  ) -> CCIResult: ...

class _cci_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for cci.

  Configuration:
      period (int)
  """

  period: int

class _cci_Config(_NCMakeableModel[_cci_Bound]):
  """Configuration class for cci.

  Calculate Commodity Channel Index (CCI).

  CCI is a momentum-based oscillator used to determine when an asset
  is overbought or oversold. It measures price variation from the mean.

  Formula:
  CCI = (Typical Price - SMA(Typical Price)) / (0.015 * Mean Deviation)

  Where:
  - Typical Price = (High + Low + Close) / 3
  - Mean Deviation = mean of absolute deviations from SMA
  - 0.015 constant scales to make 70-80% of values within +/- 100

  Interpretation:
  - CCI > +100: Overbought (potential reversal down)
  - CCI < -100: Oversold (potential reversal up)
  - CCI > 100: Overbought / strong uptrend
  - CCI < -100: Oversold / strong downtrend
  - CCI ~ 0: Ranging
  - Divergences often precede reversals

  Features:
  - Numba-optimized for performance
  - Standard constant 0.015 (Lambert's default)

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)
    constant: Scaling constant (default: 0.015)

  Returns:
    CCIResult(index, cci)

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for cci.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _cci_Bound: ...

class cci:
  Type = _cci_Bound
  Config = _cci_Config
  ConfigDict = _cci_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
    constant: float = ...,
    period: int = ...,
  ) -> CCIResult: ...

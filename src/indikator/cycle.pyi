"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import Protocol, TypedDict

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import IndicatorResult, PhasorResult, SineResult

class _ht_dcperiod_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    def __call__(self, data: Validated[pd.Series[float], Finite, NotEmpty]) -> IndicatorResult: ...

class _ht_dcperiod_ConfigDict(TypedDict, total=False):
    pass

class _ht_dcperiod_Config(_NCMakeableModel[_ht_dcperiod_Bound]):
    """Configuration class for ht_dcperiod.

    Hilbert Transform - Dominant Cycle Period.

    Args:
      data: Input price series.

    Returns:
      IndicatorResult: (Dominant Cycle Period). Use .to_pandas() for Series.
    """

    pass

class ht_dcperiod:
    Type = _ht_dcperiod_Bound
    Config = _ht_dcperiod_Config
    ConfigDict = _ht_dcperiod_ConfigDict
    def __new__(cls, data: Validated[pd.Series[float], Finite, NotEmpty]) -> IndicatorResult: ...

class _ht_dcphase_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    def __call__(self, data: Validated[pd.Series[float], Finite, NotEmpty]) -> IndicatorResult: ...

class _ht_dcphase_ConfigDict(TypedDict, total=False):
    pass

class _ht_dcphase_Config(_NCMakeableModel[_ht_dcphase_Bound]):
    """Configuration class for ht_dcphase.

    Hilbert Transform - Dominant Cycle Phase.

    Args:
      data: Input price series.

    Returns:
      IndicatorResult: (Dominant Cycle Phase). Use .to_pandas() for Series.
    """

    pass

class ht_dcphase:
    Type = _ht_dcphase_Bound
    Config = _ht_dcphase_Config
    ConfigDict = _ht_dcphase_ConfigDict
    def __new__(cls, data: Validated[pd.Series[float], Finite, NotEmpty]) -> IndicatorResult: ...

class _ht_phasor_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    def __call__(self, data: Validated[pd.Series[float], Finite, NotEmpty]) -> PhasorResult: ...

class _ht_phasor_ConfigDict(TypedDict, total=False):
    pass

class _ht_phasor_Config(_NCMakeableModel[_ht_phasor_Bound]):
    """Configuration class for ht_phasor.

    Hilbert Transform - Phasor Components.

    Args:
      data: Input price series.

    Returns:
      PhasorResult: (InPhase, Quadrature). Use .to_pandas() for DataFrame.
    """

    pass

class ht_phasor:
    Type = _ht_phasor_Bound
    Config = _ht_phasor_Config
    ConfigDict = _ht_phasor_ConfigDict
    def __new__(cls, data: Validated[pd.Series[float], Finite, NotEmpty]) -> PhasorResult: ...

class _ht_sine_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    def __call__(self, data: Validated[pd.Series[float], Finite, NotEmpty]) -> SineResult: ...

class _ht_sine_ConfigDict(TypedDict, total=False):
    pass

class _ht_sine_Config(_NCMakeableModel[_ht_sine_Bound]):
    """Configuration class for ht_sine.

    Hilbert Transform - SineWave.

    Returns the sine of the Dominant Cycle Phase and a lead sine (45 degrees advancement).

    Args:
      data: Input price series.

    Returns:
      SineResult: (Sine, LeadSine). Use .to_pandas() for DataFrame.
    """

    pass

class ht_sine:
    Type = _ht_sine_Bound
    Config = _ht_sine_Config
    ConfigDict = _ht_sine_ConfigDict
    def __new__(cls, data: Validated[pd.Series[float], Finite, NotEmpty]) -> SineResult: ...

class _ht_trendmode_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    def __call__(self, data: Validated[pd.Series[float], Finite, NotEmpty]) -> IndicatorResult: ...

class _ht_trendmode_ConfigDict(TypedDict, total=False):
    pass

class _ht_trendmode_Config(_NCMakeableModel[_ht_trendmode_Bound]):
    """Configuration class for ht_trendmode.

    Hilbert Transform - Trend vs Cycle Mode.

    Args:
      data: Input price series.

    Returns:
      IndicatorResult: TrendMode (0 or 1). Use .to_pandas() for Series.
    """

    pass

class ht_trendmode:
    Type = _ht_trendmode_Bound
    Config = _ht_trendmode_Config
    ConfigDict = _ht_trendmode_ConfigDict
    def __new__(cls, data: Validated[pd.Series[float], Finite, NotEmpty]) -> IndicatorResult: ...

class _ht_trendline_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    def __call__(self, data: Validated[pd.Series[float], Finite, NotEmpty]) -> IndicatorResult: ...

class _ht_trendline_ConfigDict(TypedDict, total=False):
    pass

class _ht_trendline_Config(_NCMakeableModel[_ht_trendline_Bound]):
    """Configuration class for ht_trendline.

    Hilbert Transform - Trendline.

    Args:
      data: Input price series.

    Returns:
      IndicatorResult: Trendline. Use .to_pandas() for Series.
    """

    pass

class ht_trendline:
    Type = _ht_trendline_Bound
    Config = _ht_trendline_Config
    ConfigDict = _ht_trendline_ConfigDict
    def __new__(cls, data: Validated[pd.Series[float], Finite, NotEmpty]) -> IndicatorResult: ...

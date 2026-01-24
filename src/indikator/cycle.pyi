"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from __future__ import annotations

from typing import Protocol, TypedDict, TYPE_CHECKING

from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

if TYPE_CHECKING:
  from datawarden import Finite, NotEmpty, Validated

class _ht_dcperiod_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(self, data: Validated[pd.Series, Finite, NotEmpty]) -> pd.Series: ...

class _ht_dcperiod_ConfigDict(TypedDict, total=False):
  pass

class _ht_dcperiod_Config(_NCMakeableModel[_ht_dcperiod_Bound]):
  """Configuration class for ht_dcperiod.

  Hilbert Transform - Dominant Cycle Period.

  Args:
    data: Input price series.

  Returns:
    pd.Series: Dominant Cycle Period.
  """

  pass

class ht_dcperiod:
  Type = _ht_dcperiod_Bound
  Config = _ht_dcperiod_Config
  ConfigDict = _ht_dcperiod_ConfigDict
  def __new__(cls, data: Validated[pd.Series, Finite, NotEmpty]) -> pd.Series: ...

class _ht_dcphase_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(self, data: Validated[pd.Series, Finite, NotEmpty]) -> pd.Series: ...

class _ht_dcphase_ConfigDict(TypedDict, total=False):
  pass

class _ht_dcphase_Config(_NCMakeableModel[_ht_dcphase_Bound]):
  """Configuration class for ht_dcphase.

  Hilbert Transform - Dominant Cycle Phase.

  Args:
    data: Input price series.

  Returns:
    pd.Series: Dominant Cycle Phase (0 to 360 degrees).
  """

  pass

class ht_dcphase:
  Type = _ht_dcphase_Bound
  Config = _ht_dcphase_Config
  ConfigDict = _ht_dcphase_ConfigDict
  def __new__(cls, data: Validated[pd.Series, Finite, NotEmpty]) -> pd.Series: ...

class _ht_phasor_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self, data: Validated[pd.Series, Finite, NotEmpty]
  ) -> tuple[pd.Series, pd.Series]: ...

class _ht_phasor_ConfigDict(TypedDict, total=False):
  pass

class _ht_phasor_Config(_NCMakeableModel[_ht_phasor_Bound]):
  """Configuration class for ht_phasor.

  Hilbert Transform - Phasor Components.

  Args:
    data: Input price series.

  Returns:
    tuple[pd.Series, pd.Series]: (InPhase, Quadrature) components.
  """

  pass

class ht_phasor:
  Type = _ht_phasor_Bound
  Config = _ht_phasor_Config
  ConfigDict = _ht_phasor_ConfigDict
  def __new__(
    cls, data: Validated[pd.Series, Finite, NotEmpty]
  ) -> tuple[pd.Series, pd.Series]: ...

class _ht_sine_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self, data: Validated[pd.Series, Finite, NotEmpty]
  ) -> tuple[pd.Series, pd.Series]: ...

class _ht_sine_ConfigDict(TypedDict, total=False):
  pass

class _ht_sine_Config(_NCMakeableModel[_ht_sine_Bound]):
  """Configuration class for ht_sine.

  Hilbert Transform - SineWave.

  Returns the sine of the Dominant Cycle Phase and a lead sine (45 degrees advancement).

  Args:
    data: Input price series.

  Returns:
    tuple[pd.Series, pd.Series]: (Sine, LeadSine).
  """

  pass

class ht_sine:
  Type = _ht_sine_Bound
  Config = _ht_sine_Config
  ConfigDict = _ht_sine_ConfigDict
  def __new__(
    cls, data: Validated[pd.Series, Finite, NotEmpty]
  ) -> tuple[pd.Series, pd.Series]: ...

class _ht_trendmode_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(self, data: Validated[pd.Series, Finite, NotEmpty]) -> pd.Series: ...

class _ht_trendmode_ConfigDict(TypedDict, total=False):
  pass

class _ht_trendmode_Config(_NCMakeableModel[_ht_trendmode_Bound]):
  """Configuration class for ht_trendmode.

  Hilbert Transform - Trend vs Cycle Mode.

  Args:
    data: Input price series.

  Returns:
    pd.Series: TrendMode (0 or 1). 1 indicates a trend is detected.
  """

  pass

class ht_trendmode:
  Type = _ht_trendmode_Bound
  Config = _ht_trendmode_Config
  ConfigDict = _ht_trendmode_ConfigDict
  def __new__(cls, data: Validated[pd.Series, Finite, NotEmpty]) -> pd.Series: ...

class _ht_trendline_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(self, data: Validated[pd.Series, Finite, NotEmpty]) -> pd.Series: ...

class _ht_trendline_ConfigDict(TypedDict, total=False):
  pass

class _ht_trendline_Config(_NCMakeableModel[_ht_trendline_Bound]):
  """Configuration class for ht_trendline.

  Hilbert Transform - Trendline.

  Args:
    data: Input price series.

  Returns:
    pd.Series: Trendline.
  """

  pass

class ht_trendline:
  Type = _ht_trendline_Bound
  Config = _ht_trendline_Config
  ConfigDict = _ht_trendline_ConfigDict
  def __new__(cls, data: Validated[pd.Series, Finite, NotEmpty]) -> pd.Series: ...

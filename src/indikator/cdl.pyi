"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import Protocol, TypedDict

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

class _cdl_doji_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_doji_ConfigDict(TypedDict, total=False):
  pass

class _cdl_doji_Config(_NCMakeableModel[_cdl_doji_Bound]):
  """Configuration class for cdl_doji.

  Detect Doji pattern.

  Returns 100 if detected, 0 otherwise.
  """

  pass

class cdl_doji:
  Type = _cdl_doji_Bound
  Config = _cdl_doji_Config
  ConfigDict = _cdl_doji_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_hammer_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_hammer_ConfigDict(TypedDict, total=False):
  pass

class _cdl_hammer_Config(_NCMakeableModel[_cdl_hammer_Bound]):
  """Configuration class for cdl_hammer.

  Detect Hammer pattern.

  Returns 100 (Bullish) if detected, 0 otherwise.
  """

  pass

class cdl_hammer:
  Type = _cdl_hammer_Bound
  Config = _cdl_hammer_Config
  ConfigDict = _cdl_hammer_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_engulfing_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_engulfing_ConfigDict(TypedDict, total=False):
  pass

class _cdl_engulfing_Config(_NCMakeableModel[_cdl_engulfing_Bound]):
  """Configuration class for cdl_engulfing.

  Detect Engulfing pattern.

  Returns:
  - 100: Bullish Engulfing
  - -100: Bearish Engulfing
  - 0: None
  """

  pass

class cdl_engulfing:
  Type = _cdl_engulfing_Bound
  Config = _cdl_engulfing_Config
  ConfigDict = _cdl_engulfing_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_harami_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_harami_ConfigDict(TypedDict, total=False):
  pass

class _cdl_harami_Config(_NCMakeableModel[_cdl_harami_Bound]):
  """Configuration class for cdl_harami.

  Detect Harami pattern.

  Returns:
  - 100: Bullish Harami
  - -100: Bearish Harami
  - 0: None
  """

  pass

class cdl_harami:
  Type = _cdl_harami_Bound
  Config = _cdl_harami_Config
  ConfigDict = _cdl_harami_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

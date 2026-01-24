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

class _cdl_shooting_star_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_shooting_star_ConfigDict(TypedDict, total=False):
  pass

class _cdl_shooting_star_Config(_NCMakeableModel[_cdl_shooting_star_Bound]):
  """Configuration class for cdl_shooting_star.

  Detect Shooting Star pattern.

  Returns:
  - -100: Bearish Shooting Star
  - 0: None
  """

  pass

class cdl_shooting_star:
  Type = _cdl_shooting_star_Bound
  Config = _cdl_shooting_star_Config
  ConfigDict = _cdl_shooting_star_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_inverted_hammer_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_inverted_hammer_ConfigDict(TypedDict, total=False):
  pass

class _cdl_inverted_hammer_Config(_NCMakeableModel[_cdl_inverted_hammer_Bound]):
  """Configuration class for cdl_inverted_hammer.

  Detect Inverted Hammer pattern.

  Returns:
  - 100: Bullish Inverted Hammer
  - 0: None
  """

  pass

class cdl_inverted_hammer:
  Type = _cdl_inverted_hammer_Bound
  Config = _cdl_inverted_hammer_Config
  ConfigDict = _cdl_inverted_hammer_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_hanging_man_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_hanging_man_ConfigDict(TypedDict, total=False):
  pass

class _cdl_hanging_man_Config(_NCMakeableModel[_cdl_hanging_man_Bound]):
  """Configuration class for cdl_hanging_man.

  Detect Hanging Man pattern.

  Returns:
  - -100: Bearish Hanging Man
  - 0: None
  """

  pass

class cdl_hanging_man:
  Type = _cdl_hanging_man_Bound
  Config = _cdl_hanging_man_Config
  ConfigDict = _cdl_hanging_man_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_marubozu_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_marubozu_ConfigDict(TypedDict, total=False):
  pass

class _cdl_marubozu_Config(_NCMakeableModel[_cdl_marubozu_Bound]):
  """Configuration class for cdl_marubozu.

  Detect Marubozu pattern.

  Returns:
  - 100: Bullish (White) Marubozu
  - -100: Bearish (Black) Marubozu
  - 0: None
  """

  pass

class cdl_marubozu:
  Type = _cdl_marubozu_Bound
  Config = _cdl_marubozu_Config
  ConfigDict = _cdl_marubozu_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_morning_star_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_morning_star_ConfigDict(TypedDict, total=False):
  pass

class _cdl_morning_star_Config(_NCMakeableModel[_cdl_morning_star_Bound]):
  """Configuration class for cdl_morning_star.

  Detect Morning Star pattern.

  Returns:
  - 100: Bullish Morning Star
  - 0: None
  """

  pass

class cdl_morning_star:
  Type = _cdl_morning_star_Bound
  Config = _cdl_morning_star_Config
  ConfigDict = _cdl_morning_star_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_evening_star_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_evening_star_ConfigDict(TypedDict, total=False):
  pass

class _cdl_evening_star_Config(_NCMakeableModel[_cdl_evening_star_Bound]):
  """Configuration class for cdl_evening_star.

  Detect Evening Star pattern.

  Returns:
  - -100: Bearish Evening Star
  - 0: None
  """

  pass

class cdl_evening_star:
  Type = _cdl_evening_star_Bound
  Config = _cdl_evening_star_Config
  ConfigDict = _cdl_evening_star_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_3black_crows_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_3black_crows_ConfigDict(TypedDict, total=False):
  pass

class _cdl_3black_crows_Config(_NCMakeableModel[_cdl_3black_crows_Bound]):
  """Configuration class for cdl_3black_crows.

  Detect Three Black Crows pattern.

  Returns:
  - -100: Bearish Three Black Crows
  - 0: None
  """

  pass

class cdl_3black_crows:
  Type = _cdl_3black_crows_Bound
  Config = _cdl_3black_crows_Config
  ConfigDict = _cdl_3black_crows_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_3white_soldiers_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_3white_soldiers_ConfigDict(TypedDict, total=False):
  pass

class _cdl_3white_soldiers_Config(_NCMakeableModel[_cdl_3white_soldiers_Bound]):
  """Configuration class for cdl_3white_soldiers.

  Detect Three White Soldiers pattern.

  Returns:
  - 100: Bullish Three White Soldiers
  - 0: None
  """

  pass

class cdl_3white_soldiers:
  Type = _cdl_3white_soldiers_Bound
  Config = _cdl_3white_soldiers_Config
  ConfigDict = _cdl_3white_soldiers_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_3inside_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_3inside_ConfigDict(TypedDict, total=False):
  pass

class _cdl_3inside_Config(_NCMakeableModel[_cdl_3inside_Bound]):
  """Configuration class for cdl_3inside.

  Detect Three Inside Up/Down pattern.

  Returns:
  - 100: Three Inside Up (Bullish)
  - -100: Three Inside Down (Bearish)
  """

  pass

class cdl_3inside:
  Type = _cdl_3inside_Bound
  Config = _cdl_3inside_Config
  ConfigDict = _cdl_3inside_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_3outside_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_3outside_ConfigDict(TypedDict, total=False):
  pass

class _cdl_3outside_Config(_NCMakeableModel[_cdl_3outside_Bound]):
  """Configuration class for cdl_3outside.

  Detect Three Outside Up/Down pattern.

  Returns:
  - 100: Three Outside Up (Bullish)
  - -100: Three Outside Down (Bearish)
  """

  pass

class cdl_3outside:
  Type = _cdl_3outside_Bound
  Config = _cdl_3outside_Config
  ConfigDict = _cdl_3outside_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_3line_strike_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_3line_strike_ConfigDict(TypedDict, total=False):
  pass

class _cdl_3line_strike_Config(_NCMakeableModel[_cdl_3line_strike_Bound]):
  """Configuration class for cdl_3line_strike.

  Detect Three Line Strike pattern.

  Returns:
  - 100: Bullish Strike
  - -100: Bearish Strike
  """

  pass

class cdl_3line_strike:
  Type = _cdl_3line_strike_Bound
  Config = _cdl_3line_strike_Config
  ConfigDict = _cdl_3line_strike_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_piercing_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_piercing_ConfigDict(TypedDict, total=False):
  pass

class _cdl_piercing_Config(_NCMakeableModel[_cdl_piercing_Bound]):
  """Configuration class for cdl_piercing.

  Detect Piercing Pattern.

  Returns:
  - 100: Bullish Piercing
  - 0: None
  """

  pass

class cdl_piercing:
  Type = _cdl_piercing_Bound
  Config = _cdl_piercing_Config
  ConfigDict = _cdl_piercing_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_dark_cloud_cover_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_dark_cloud_cover_ConfigDict(TypedDict, total=False):
  pass

class _cdl_dark_cloud_cover_Config(_NCMakeableModel[_cdl_dark_cloud_cover_Bound]):
  """Configuration class for cdl_dark_cloud_cover.

  Detect Dark Cloud Cover Pattern.

  Returns:
  - -100: Bearish Dark Cloud Cover
  - 0: None
  """

  pass

class cdl_dark_cloud_cover:
  Type = _cdl_dark_cloud_cover_Bound
  Config = _cdl_dark_cloud_cover_Config
  ConfigDict = _cdl_dark_cloud_cover_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_kicking_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_kicking_ConfigDict(TypedDict, total=False):
  pass

class _cdl_kicking_Config(_NCMakeableModel[_cdl_kicking_Bound]):
  """Configuration class for cdl_kicking.

  Detect Kicking Pattern.

  Returns:
  - 100: Bullish Kicking
  - -100: Bearish Kicking
  """

  pass

class cdl_kicking:
  Type = _cdl_kicking_Bound
  Config = _cdl_kicking_Config
  ConfigDict = _cdl_kicking_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_matching_low_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_matching_low_ConfigDict(TypedDict, total=False):
  pass

class _cdl_matching_low_Config(_NCMakeableModel[_cdl_matching_low_Bound]):
  """Configuration class for cdl_matching_low.

  Detect Matching Low Pattern.

  Returns:
  - 100: Bullish Matching Low
  - 0: None
  """

  pass

class cdl_matching_low:
  Type = _cdl_matching_low_Bound
  Config = _cdl_matching_low_Config
  ConfigDict = _cdl_matching_low_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_spinning_top_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_spinning_top_ConfigDict(TypedDict, total=False):
  pass

class _cdl_spinning_top_Config(_NCMakeableModel[_cdl_spinning_top_Bound]):
  """Configuration class for cdl_spinning_top.

  Detect Spinning Top pattern.

  Returns:
  - 100: Bullish/Neutral Spinning Top
  - -100: Bearish/Neutral Spinning Top
  """

  pass

class cdl_spinning_top:
  Type = _cdl_spinning_top_Bound
  Config = _cdl_spinning_top_Config
  ConfigDict = _cdl_spinning_top_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_rickshaw_man_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_rickshaw_man_ConfigDict(TypedDict, total=False):
  pass

class _cdl_rickshaw_man_Config(_NCMakeableModel[_cdl_rickshaw_man_Bound]):
  """Configuration class for cdl_rickshaw_man.

  Detect Rickshaw Man pattern.

  Returns:
  - 100: Detected
  """

  pass

class cdl_rickshaw_man:
  Type = _cdl_rickshaw_man_Bound
  Config = _cdl_rickshaw_man_Config
  ConfigDict = _cdl_rickshaw_man_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_high_wave_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_high_wave_ConfigDict(TypedDict, total=False):
  pass

class _cdl_high_wave_Config(_NCMakeableModel[_cdl_high_wave_Bound]):
  """Configuration class for cdl_high_wave.

  Detect High Wave pattern.

  Returns:
  - 100: Bullish High Wave
  - -100: Bearish High Wave
  """

  pass

class cdl_high_wave:
  Type = _cdl_high_wave_Bound
  Config = _cdl_high_wave_Config
  ConfigDict = _cdl_high_wave_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_long_legged_doji_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_long_legged_doji_ConfigDict(TypedDict, total=False):
  pass

class _cdl_long_legged_doji_Config(_NCMakeableModel[_cdl_long_legged_doji_Bound]):
  """Configuration class for cdl_long_legged_doji.

  Detect Long Legged Doji pattern.

  Returns:
  - 100: Detected
  """

  pass

class cdl_long_legged_doji:
  Type = _cdl_long_legged_doji_Bound
  Config = _cdl_long_legged_doji_Config
  ConfigDict = _cdl_long_legged_doji_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_tristar_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_tristar_ConfigDict(TypedDict, total=False):
  pass

class _cdl_tristar_Config(_NCMakeableModel[_cdl_tristar_Bound]):
  """Configuration class for cdl_tristar.

  Detect Tristar pattern.

  Returns:
  - 100: Bullish Tristar
  - -100: Bearish Tristar
  """

  pass

class cdl_tristar:
  Type = _cdl_tristar_Bound
  Config = _cdl_tristar_Config
  ConfigDict = _cdl_tristar_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_tasuki_gap_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_tasuki_gap_ConfigDict(TypedDict, total=False):
  pass

class _cdl_tasuki_gap_Config(_NCMakeableModel[_cdl_tasuki_gap_Bound]):
  """Configuration class for cdl_tasuki_gap.

  Detect Tasuki Gap pattern.

  Returns:
  - 100: Upside Tasuki Gap
  - -100: Downside Tasuki Gap
  """

  pass

class cdl_tasuki_gap:
  Type = _cdl_tasuki_gap_Bound
  Config = _cdl_tasuki_gap_Config
  ConfigDict = _cdl_tasuki_gap_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_separating_lines_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_separating_lines_ConfigDict(TypedDict, total=False):
  pass

class _cdl_separating_lines_Config(_NCMakeableModel[_cdl_separating_lines_Bound]):
  """Configuration class for cdl_separating_lines.

  Detect Separating Lines pattern.

  Returns:
  - 100: Bullish
  - -100: Bearish
  """

  pass

class cdl_separating_lines:
  Type = _cdl_separating_lines_Bound
  Config = _cdl_separating_lines_Config
  ConfigDict = _cdl_separating_lines_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_gap_side_by_side_white_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_gap_side_by_side_white_ConfigDict(TypedDict, total=False):
  pass

class _cdl_gap_side_by_side_white_Config(
  _NCMakeableModel[_cdl_gap_side_by_side_white_Bound]
):
  """Configuration class for cdl_gap_side_by_side_white.

  Detect Gap Side-by-Side White Lines.

  Returns:
  - 100: Bullish
  - -100: Bearish
  """

  pass

class cdl_gap_side_by_side_white:
  Type = _cdl_gap_side_by_side_white_Bound
  Config = _cdl_gap_side_by_side_white_Config
  ConfigDict = _cdl_gap_side_by_side_white_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_2crows_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_2crows_ConfigDict(TypedDict, total=False):
  pass

class _cdl_2crows_Config(_NCMakeableModel[_cdl_2crows_Bound]):
  """Configuration class for cdl_2crows.

  Detect Two Crows pattern.

  Returns:
  - -100: Bearish Two Crows
  """

  pass

class cdl_2crows:
  Type = _cdl_2crows_Bound
  Config = _cdl_2crows_Config
  ConfigDict = _cdl_2crows_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_upside_gap_two_crows_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_upside_gap_two_crows_ConfigDict(TypedDict, total=False):
  pass

class _cdl_upside_gap_two_crows_Config(
  _NCMakeableModel[_cdl_upside_gap_two_crows_Bound]
):
  """Configuration class for cdl_upside_gap_two_crows.

  Detect Upside Gap Two Crows pattern.

  Returns:
  - -100: Bearish Upside Gap Two Crows
  """

  pass

class cdl_upside_gap_two_crows:
  Type = _cdl_upside_gap_two_crows_Bound
  Config = _cdl_upside_gap_two_crows_Config
  ConfigDict = _cdl_upside_gap_two_crows_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_abandoned_baby_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_abandoned_baby_ConfigDict(TypedDict, total=False):
  pass

class _cdl_abandoned_baby_Config(_NCMakeableModel[_cdl_abandoned_baby_Bound]):
  """Configuration class for cdl_abandoned_baby.

  Detect Abandoned Baby.
  """

  pass

class cdl_abandoned_baby:
  Type = _cdl_abandoned_baby_Bound
  Config = _cdl_abandoned_baby_Config
  ConfigDict = _cdl_abandoned_baby_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_advance_block_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_advance_block_ConfigDict(TypedDict, total=False):
  pass

class _cdl_advance_block_Config(_NCMakeableModel[_cdl_advance_block_Bound]):
  """Configuration class for cdl_advance_block.

  Detect Advance Block.
  """

  pass

class cdl_advance_block:
  Type = _cdl_advance_block_Bound
  Config = _cdl_advance_block_Config
  ConfigDict = _cdl_advance_block_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_belt_hold_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_belt_hold_ConfigDict(TypedDict, total=False):
  pass

class _cdl_belt_hold_Config(_NCMakeableModel[_cdl_belt_hold_Bound]):
  """Configuration class for cdl_belt_hold.

  Detect Belt Hold.
  """

  pass

class cdl_belt_hold:
  Type = _cdl_belt_hold_Bound
  Config = _cdl_belt_hold_Config
  ConfigDict = _cdl_belt_hold_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_breakaway_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_breakaway_ConfigDict(TypedDict, total=False):
  pass

class _cdl_breakaway_Config(_NCMakeableModel[_cdl_breakaway_Bound]):
  """Configuration class for cdl_breakaway.

  Detect Breakaway.
  """

  pass

class cdl_breakaway:
  Type = _cdl_breakaway_Bound
  Config = _cdl_breakaway_Config
  ConfigDict = _cdl_breakaway_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_closing_marubozu_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_closing_marubozu_ConfigDict(TypedDict, total=False):
  pass

class _cdl_closing_marubozu_Config(_NCMakeableModel[_cdl_closing_marubozu_Bound]):
  """Configuration class for cdl_closing_marubozu.

  Detect Closing Marubozu.
  """

  pass

class cdl_closing_marubozu:
  Type = _cdl_closing_marubozu_Bound
  Config = _cdl_closing_marubozu_Config
  ConfigDict = _cdl_closing_marubozu_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_dragonfly_doji_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_dragonfly_doji_ConfigDict(TypedDict, total=False):
  pass

class _cdl_dragonfly_doji_Config(_NCMakeableModel[_cdl_dragonfly_doji_Bound]):
  """Configuration class for cdl_dragonfly_doji.

  Detect Dragonfly Doji.
  """

  pass

class cdl_dragonfly_doji:
  Type = _cdl_dragonfly_doji_Bound
  Config = _cdl_dragonfly_doji_Config
  ConfigDict = _cdl_dragonfly_doji_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_gravestone_doji_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_gravestone_doji_ConfigDict(TypedDict, total=False):
  pass

class _cdl_gravestone_doji_Config(_NCMakeableModel[_cdl_gravestone_doji_Bound]):
  """Configuration class for cdl_gravestone_doji.

  Detect Gravestone Doji.
  """

  pass

class cdl_gravestone_doji:
  Type = _cdl_gravestone_doji_Bound
  Config = _cdl_gravestone_doji_Config
  ConfigDict = _cdl_gravestone_doji_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_hikkake_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_hikkake_ConfigDict(TypedDict, total=False):
  pass

class _cdl_hikkake_Config(_NCMakeableModel[_cdl_hikkake_Bound]):
  """Configuration class for cdl_hikkake.

  Detect Hikkake.
  """

  pass

class cdl_hikkake:
  Type = _cdl_hikkake_Bound
  Config = _cdl_hikkake_Config
  ConfigDict = _cdl_hikkake_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_homing_pigeon_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_homing_pigeon_ConfigDict(TypedDict, total=False):
  pass

class _cdl_homing_pigeon_Config(_NCMakeableModel[_cdl_homing_pigeon_Bound]):
  """Configuration class for cdl_homing_pigeon.

  Detect Homing Pigeon.
  """

  pass

class cdl_homing_pigeon:
  Type = _cdl_homing_pigeon_Bound
  Config = _cdl_homing_pigeon_Config
  ConfigDict = _cdl_homing_pigeon_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_identical_3crows_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_identical_3crows_ConfigDict(TypedDict, total=False):
  pass

class _cdl_identical_3crows_Config(_NCMakeableModel[_cdl_identical_3crows_Bound]):
  """Configuration class for cdl_identical_3crows.

  Detect Identical Three Crows.
  """

  pass

class cdl_identical_3crows:
  Type = _cdl_identical_3crows_Bound
  Config = _cdl_identical_3crows_Config
  ConfigDict = _cdl_identical_3crows_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_in_neck_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_in_neck_ConfigDict(TypedDict, total=False):
  pass

class _cdl_in_neck_Config(_NCMakeableModel[_cdl_in_neck_Bound]):
  """Configuration class for cdl_in_neck.

  Detect In-Neck.
  """

  pass

class cdl_in_neck:
  Type = _cdl_in_neck_Bound
  Config = _cdl_in_neck_Config
  ConfigDict = _cdl_in_neck_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_ladder_bottom_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_ladder_bottom_ConfigDict(TypedDict, total=False):
  pass

class _cdl_ladder_bottom_Config(_NCMakeableModel[_cdl_ladder_bottom_Bound]):
  """Configuration class for cdl_ladder_bottom.

  Detect Ladder Bottom.
  """

  pass

class cdl_ladder_bottom:
  Type = _cdl_ladder_bottom_Bound
  Config = _cdl_ladder_bottom_Config
  ConfigDict = _cdl_ladder_bottom_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_long_line_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_long_line_ConfigDict(TypedDict, total=False):
  pass

class _cdl_long_line_Config(_NCMakeableModel[_cdl_long_line_Bound]):
  """Configuration class for cdl_long_line.

  Detect Long Line.
  """

  pass

class cdl_long_line:
  Type = _cdl_long_line_Bound
  Config = _cdl_long_line_Config
  ConfigDict = _cdl_long_line_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_mat_hold_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_mat_hold_ConfigDict(TypedDict, total=False):
  pass

class _cdl_mat_hold_Config(_NCMakeableModel[_cdl_mat_hold_Bound]):
  """Configuration class for cdl_mat_hold.

  Detect Mat Hold.
  """

  pass

class cdl_mat_hold:
  Type = _cdl_mat_hold_Bound
  Config = _cdl_mat_hold_Config
  ConfigDict = _cdl_mat_hold_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_on_neck_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_on_neck_ConfigDict(TypedDict, total=False):
  pass

class _cdl_on_neck_Config(_NCMakeableModel[_cdl_on_neck_Bound]):
  """Configuration class for cdl_on_neck.

  Detect On-Neck.
  """

  pass

class cdl_on_neck:
  Type = _cdl_on_neck_Bound
  Config = _cdl_on_neck_Config
  ConfigDict = _cdl_on_neck_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_rise_fall_3methods_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_rise_fall_3methods_ConfigDict(TypedDict, total=False):
  pass

class _cdl_rise_fall_3methods_Config(_NCMakeableModel[_cdl_rise_fall_3methods_Bound]):
  """Configuration class for cdl_rise_fall_3methods.

  Detect Rise/Fall Three Methods.
  """

  pass

class cdl_rise_fall_3methods:
  Type = _cdl_rise_fall_3methods_Bound
  Config = _cdl_rise_fall_3methods_Config
  ConfigDict = _cdl_rise_fall_3methods_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_short_line_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_short_line_ConfigDict(TypedDict, total=False):
  pass

class _cdl_short_line_Config(_NCMakeableModel[_cdl_short_line_Bound]):
  """Configuration class for cdl_short_line.

  Detect Short Line.
  """

  pass

class cdl_short_line:
  Type = _cdl_short_line_Bound
  Config = _cdl_short_line_Config
  ConfigDict = _cdl_short_line_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_stalled_pattern_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_stalled_pattern_ConfigDict(TypedDict, total=False):
  pass

class _cdl_stalled_pattern_Config(_NCMakeableModel[_cdl_stalled_pattern_Bound]):
  """Configuration class for cdl_stalled_pattern.

  Detect Stalled Pattern.
  """

  pass

class cdl_stalled_pattern:
  Type = _cdl_stalled_pattern_Bound
  Config = _cdl_stalled_pattern_Config
  ConfigDict = _cdl_stalled_pattern_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_stick_sandwich_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_stick_sandwich_ConfigDict(TypedDict, total=False):
  pass

class _cdl_stick_sandwich_Config(_NCMakeableModel[_cdl_stick_sandwich_Bound]):
  """Configuration class for cdl_stick_sandwich.

  Detect Stick Sandwich.
  """

  pass

class cdl_stick_sandwich:
  Type = _cdl_stick_sandwich_Bound
  Config = _cdl_stick_sandwich_Config
  ConfigDict = _cdl_stick_sandwich_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_takuri_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_takuri_ConfigDict(TypedDict, total=False):
  pass

class _cdl_takuri_Config(_NCMakeableModel[_cdl_takuri_Bound]):
  """Configuration class for cdl_takuri.

  Detect Takuri.
  """

  pass

class cdl_takuri:
  Type = _cdl_takuri_Bound
  Config = _cdl_takuri_Config
  ConfigDict = _cdl_takuri_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_thrusting_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_thrusting_ConfigDict(TypedDict, total=False):
  pass

class _cdl_thrusting_Config(_NCMakeableModel[_cdl_thrusting_Bound]):
  """Configuration class for cdl_thrusting.

  Detect Thrusting.
  """

  pass

class cdl_thrusting:
  Type = _cdl_thrusting_Bound
  Config = _cdl_thrusting_Config
  ConfigDict = _cdl_thrusting_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_unique_3river_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_unique_3river_ConfigDict(TypedDict, total=False):
  pass

class _cdl_unique_3river_Config(_NCMakeableModel[_cdl_unique_3river_Bound]):
  """Configuration class for cdl_unique_3river.

  Detect Unique 3 River.
  """

  pass

class cdl_unique_3river:
  Type = _cdl_unique_3river_Bound
  Config = _cdl_unique_3river_Config
  ConfigDict = _cdl_unique_3river_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_counterattack_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_counterattack_ConfigDict(TypedDict, total=False):
  pass

class _cdl_counterattack_Config(_NCMakeableModel[_cdl_counterattack_Bound]):
  """Configuration class for cdl_counterattack.

  Detect Counterattack.
  """

  pass

class cdl_counterattack:
  Type = _cdl_counterattack_Bound
  Config = _cdl_counterattack_Config
  ConfigDict = _cdl_counterattack_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_doji_star_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_doji_star_ConfigDict(TypedDict, total=False):
  pass

class _cdl_doji_star_Config(_NCMakeableModel[_cdl_doji_star_Bound]):
  """Configuration class for cdl_doji_star.

  Detect Doji Star.
  """

  pass

class cdl_doji_star:
  Type = _cdl_doji_star_Bound
  Config = _cdl_doji_star_Config
  ConfigDict = _cdl_doji_star_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_conceal_baby_swallow_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_conceal_baby_swallow_ConfigDict(TypedDict, total=False):
  pass

class _cdl_conceal_baby_swallow_Config(
  _NCMakeableModel[_cdl_conceal_baby_swallow_Bound]
):
  """Configuration class for cdl_conceal_baby_swallow.

  Detect Concealing Baby Swallow.
  """

  pass

class cdl_conceal_baby_swallow:
  Type = _cdl_conceal_baby_swallow_Bound
  Config = _cdl_conceal_baby_swallow_Config
  ConfigDict = _cdl_conceal_baby_swallow_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_harami_cross_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_harami_cross_ConfigDict(TypedDict, total=False):
  pass

class _cdl_harami_cross_Config(_NCMakeableModel[_cdl_harami_cross_Bound]):
  """Configuration class for cdl_harami_cross.

  Detect Harami Cross.
  """

  pass

class cdl_harami_cross:
  Type = _cdl_harami_cross_Bound
  Config = _cdl_harami_cross_Config
  ConfigDict = _cdl_harami_cross_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_hikkake_mod_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_hikkake_mod_ConfigDict(TypedDict, total=False):
  pass

class _cdl_hikkake_mod_Config(_NCMakeableModel[_cdl_hikkake_mod_Bound]):
  """Configuration class for cdl_hikkake_mod.

  Detect Modified Hikkake.
  """

  pass

class cdl_hikkake_mod:
  Type = _cdl_hikkake_mod_Bound
  Config = _cdl_hikkake_mod_Config
  ConfigDict = _cdl_hikkake_mod_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_morning_doji_star_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_morning_doji_star_ConfigDict(TypedDict, total=False):
  pass

class _cdl_morning_doji_star_Config(_NCMakeableModel[_cdl_morning_doji_star_Bound]):
  """Configuration class for cdl_morning_doji_star.

  Detect Morning Doji Star.
  """

  pass

class cdl_morning_doji_star:
  Type = _cdl_morning_doji_star_Bound
  Config = _cdl_morning_doji_star_Config
  ConfigDict = _cdl_morning_doji_star_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_evening_doji_star_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_evening_doji_star_ConfigDict(TypedDict, total=False):
  pass

class _cdl_evening_doji_star_Config(_NCMakeableModel[_cdl_evening_doji_star_Bound]):
  """Configuration class for cdl_evening_doji_star.

  Detect Evening Doji Star.
  """

  pass

class cdl_evening_doji_star:
  Type = _cdl_evening_doji_star_Bound
  Config = _cdl_evening_doji_star_Config
  ConfigDict = _cdl_evening_doji_star_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_kicking_by_length_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_kicking_by_length_ConfigDict(TypedDict, total=False):
  pass

class _cdl_kicking_by_length_Config(_NCMakeableModel[_cdl_kicking_by_length_Bound]):
  """Configuration class for cdl_kicking_by_length.

  Detect Kicking By Length.
  """

  pass

class cdl_kicking_by_length:
  Type = _cdl_kicking_by_length_Bound
  Config = _cdl_kicking_by_length_Config
  ConfigDict = _cdl_kicking_by_length_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_3stars_in_south_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_3stars_in_south_ConfigDict(TypedDict, total=False):
  pass

class _cdl_3stars_in_south_Config(_NCMakeableModel[_cdl_3stars_in_south_Bound]):
  """Configuration class for cdl_3stars_in_south.

  Detect Three Stars In The South.
  """

  pass

class cdl_3stars_in_south:
  Type = _cdl_3stars_in_south_Bound
  Config = _cdl_3stars_in_south_Config
  ConfigDict = _cdl_3stars_in_south_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_xsidegap3methods_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _cdl_xsidegap3methods_ConfigDict(TypedDict, total=False):
  pass

class _cdl_xsidegap3methods_Config(_NCMakeableModel[_cdl_xsidegap3methods_Bound]):
  """Configuration class for cdl_xsidegap3methods.

  Detect Upside/Downside Gap Three Methods.

  Returns:
  - 100: Bullish
  - -100: Bearish
  """

  pass

class cdl_xsidegap3methods:
  Type = _cdl_xsidegap3methods_Bound
  Config = _cdl_xsidegap3methods_Config
  ConfigDict = _cdl_xsidegap3methods_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

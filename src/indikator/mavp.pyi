"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Protocol, TYPE_CHECKING, TypedDict

from nonfig import Ge, Hyper, MakeableModel as _NCMakeableModel
import pandas as pd

if TYPE_CHECKING:
  from datawarden import Finite, NotEmpty, Validated

class _mavp_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    data: Validated[pd.Series, Finite, NotEmpty],
    periods: Validated[pd.Series, Finite, NotEmpty],
    minperiod: Annotated[int, Hyper(Ge(2))] = ...,
    maxperiod: Annotated[int, Hyper(Ge(2))] = ...,
    matype: int = ...,
  ) -> pd.Series: ...

class _mavp_ConfigDict(TypedDict, total=False):
  pass

class _mavp_Config(_NCMakeableModel[_mavp_Bound]):
  """Configuration class for mavp.

  Moving Average with Variable Period.

  Calculates a moving average where the period varies per element.
  Currently supports SMA (Simple Moving Average) logic (matype=0).

  Args:
    data: Input price series.
    periods: Series containing the period to use for each element.
             Values are clamped between minperiod and maxperiod.
    minperiod: Minimum period allowed (default: 2).
    maxperiod: Maximum period allowed (default: 30).
    matype: Moving Average Type (0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA).

  Returns:
    pd.Series: MAVP values.
  """

  pass

class mavp:
  Type = _mavp_Bound
  Config = _mavp_Config
  ConfigDict = _mavp_ConfigDict
  def __new__(
    cls,
    data: Validated[pd.Series, Finite, NotEmpty],
    periods: Validated[pd.Series, Finite, NotEmpty],
    minperiod: Annotated[int, Hyper(Ge(2))] = ...,
    maxperiod: Annotated[int, Hyper(Ge(2))] = ...,
    matype: int = ...,
  ) -> pd.Series: ...

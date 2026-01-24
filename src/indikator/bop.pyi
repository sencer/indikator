"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from __future__ import annotations

from typing import Protocol, TypedDict, TYPE_CHECKING

from nonfig import MakeableModel as _NCMakeableModel

from indikator._results import BOPResult

if TYPE_CHECKING:
  from datawarden import Finite, NotEmpty, Validated
  import pandas as pd

class _bop_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> BOPResult: ...

class _bop_ConfigDict(TypedDict, total=False):
  pass

class _bop_Config(_NCMakeableModel[_bop_Bound]):
  """Configuration class for bop.

  Balance of Power (BOP).

  BOP = (Close - Open) / (High - Low)

  Args:
      open_: Open prices
      high: High prices
      low: Low prices
      close: Close prices

  Returns:
      BOPResult: Balance of Power values
  """

  pass

class bop:
  Type = _bop_Bound
  Config = _bop_Config
  ConfigDict = _bop_ConfigDict
  def __new__(
    cls,
    open_: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> BOPResult: ...

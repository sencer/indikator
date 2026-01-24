"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from __future__ import annotations

from typing import ClassVar, Protocol, TypedDict, override

from nonfig import MakeableModel as _NCMakeableModel

from indikator._results import MIDPOINTResult

class _midpoint_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def period(self) -> int: ...
    def __call__(self, data: Validated[pd.Series, Finite, NotEmpty]) -> MIDPOINTResult: ...

class _midpoint_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for midpoint.

    Configuration:
        period (int)
    """

    period: int

class _midpoint_Config(_NCMakeableModel[_midpoint_Bound]):
    """Configuration class for midpoint.

    Calculate Midpoint over period.

    MIDPOINT = (highest value + lowest value) / 2

    Args:
      data: Input prices
      period: Lookback period (default 14)

    Returns:
      MIDPOINTResult

    Configuration:
        period (int)
    """

    period: int
    def __init__(self, *, period: int = ...) -> None: ...
    """Initialize configuration for midpoint.

    Configuration:
        period (int)
    """

    @override
    def make(self) -> _midpoint_Bound: ...

class midpoint:
    Type = _midpoint_Bound
    Config = _midpoint_Config
    ConfigDict = _midpoint_ConfigDict
    period: ClassVar[int]
    def __new__(cls, data: Validated[pd.Series, Finite, NotEmpty], period: int = ...) -> MIDPOINTResult: ...

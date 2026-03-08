"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import IndicatorResult

class _roc_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def period(self) -> int: ...
    def __call__(self, data: Validated[pd.Series[float], Finite, NotEmpty]) -> IndicatorResult: ...

class _roc_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for roc.

    Configuration:
        period (int)
    """

    period: int

class _roc_Config(_NCMakeableModel[_roc_Bound]):
    """Configuration class for roc.

    Calculate Rate of Change (ROC).

    ROC = ((Price - Prev) / Prev) * 100

    Configuration:
        period (int)
    """

    period: int
    def __init__(self, *, period: int = ...) -> None: ...
    """Initialize configuration for roc.

    Configuration:
        period (int)
    """

    @override
    def make(self) -> _roc_Bound: ...

class roc:
    Type = _roc_Bound
    Config = _roc_Config
    ConfigDict = _roc_ConfigDict
    period: ClassVar[int]
    def __new__(cls, data: Validated[pd.Series[float], Finite, NotEmpty], period: int = ...) -> IndicatorResult: ...

class _rocp_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def period(self) -> int: ...
    def __call__(self, data: Validated[pd.Series[float], Finite, NotEmpty]) -> IndicatorResult: ...

class _rocp_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for rocp.

    Configuration:
        period (int)
    """

    period: int

class _rocp_Config(_NCMakeableModel[_rocp_Bound]):
    """Configuration class for rocp.

    Calculate Rate of Change Percentage (ROCP).

    ROCP = (Price - Prev) / Prev

    Configuration:
        period (int)
    """

    period: int
    def __init__(self, *, period: int = ...) -> None: ...
    """Initialize configuration for rocp.

    Configuration:
        period (int)
    """

    @override
    def make(self) -> _rocp_Bound: ...

class rocp:
    Type = _rocp_Bound
    Config = _rocp_Config
    ConfigDict = _rocp_ConfigDict
    period: ClassVar[int]
    def __new__(cls, data: Validated[pd.Series[float], Finite, NotEmpty], period: int = ...) -> IndicatorResult: ...

class _rocr_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def period(self) -> int: ...
    def __call__(self, data: Validated[pd.Series[float], Finite, NotEmpty]) -> IndicatorResult: ...

class _rocr_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for rocr.

    Configuration:
        period (int)
    """

    period: int

class _rocr_Config(_NCMakeableModel[_rocr_Bound]):
    """Configuration class for rocr.

    Calculate Rate of Change Ratio (ROCR).

    ROCR = Price / Prev

    Configuration:
        period (int)
    """

    period: int
    def __init__(self, *, period: int = ...) -> None: ...
    """Initialize configuration for rocr.

    Configuration:
        period (int)
    """

    @override
    def make(self) -> _rocr_Bound: ...

class rocr:
    Type = _rocr_Bound
    Config = _rocr_Config
    ConfigDict = _rocr_ConfigDict
    period: ClassVar[int]
    def __new__(cls, data: Validated[pd.Series[float], Finite, NotEmpty], period: int = ...) -> IndicatorResult: ...

class _rocr100_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def period(self) -> int: ...
    def __call__(self, data: Validated[pd.Series[float], Finite, NotEmpty]) -> IndicatorResult: ...

class _rocr100_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for rocr100.

    Configuration:
        period (int)
    """

    period: int

class _rocr100_Config(_NCMakeableModel[_rocr100_Bound]):
    """Configuration class for rocr100.

    Calculate Rate of Change Ratio 100 Scale (ROCR100).

    ROCR100 = (Price / Prev) * 100

    Configuration:
        period (int)
    """

    period: int
    def __init__(self, *, period: int = ...) -> None: ...
    """Initialize configuration for rocr100.

    Configuration:
        period (int)
    """

    @override
    def make(self) -> _rocr100_Bound: ...

class rocr100:
    Type = _rocr100_Bound
    Config = _rocr100_Config
    ConfigDict = _rocr100_ConfigDict
    period: ClassVar[int]
    def __new__(cls, data: Validated[pd.Series[float], Finite, NotEmpty], period: int = ...) -> IndicatorResult: ...

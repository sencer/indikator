"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from __future__ import annotations

from typing import ClassVar, Protocol, TypedDict, override

from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

class _min_val_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def period(self) -> int: ...
    def __call__(self, data: Validated[pd.Series, Finite, NotEmpty]) -> pd.Series: ...

class _min_val_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for min_val.

    Configuration:
        period (int)
    """

    period: int

class _min_val_Config(_NCMakeableModel[_min_val_Bound]):
    """Configuration class for min_val.

    Rolling Minimum of a series.

    Args:
      data: Input series used for min calculation.
      period: The lookback window size (default: 30).

    Returns:
      pd.Series: Rolling minimum values.

    Configuration:
        period (int)
    """

    period: int
    def __init__(self, *, period: int = ...) -> None: ...
    """Initialize configuration for min_val.

    Configuration:
        period (int)
    """

    @override
    def make(self) -> _min_val_Bound: ...

class min_val:
    Type = _min_val_Bound
    Config = _min_val_Config
    ConfigDict = _min_val_ConfigDict
    period: ClassVar[int]
    def __new__(cls, data: Validated[pd.Series, Finite, NotEmpty], period: int = ...) -> pd.Series: ...

class _max_val_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def period(self) -> int: ...
    def __call__(self, data: Validated[pd.Series, Finite, NotEmpty]) -> pd.Series: ...

class _max_val_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for max_val.

    Configuration:
        period (int)
    """

    period: int

class _max_val_Config(_NCMakeableModel[_max_val_Bound]):
    """Configuration class for max_val.

    Rolling Maximum of a series.

    Args:
      data: Input series used for max calculation.
      period: The lookback window size (default: 30).

    Returns:
      pd.Series: Rolling maximum values.

    Configuration:
        period (int)
    """

    period: int
    def __init__(self, *, period: int = ...) -> None: ...
    """Initialize configuration for max_val.

    Configuration:
        period (int)
    """

    @override
    def make(self) -> _max_val_Bound: ...

class max_val:
    Type = _max_val_Bound
    Config = _max_val_Config
    ConfigDict = _max_val_ConfigDict
    period: ClassVar[int]
    def __new__(cls, data: Validated[pd.Series, Finite, NotEmpty], period: int = ...) -> pd.Series: ...

class _min_index_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def period(self) -> int: ...
    def __call__(self, data: Validated[pd.Series, Finite, NotEmpty]) -> pd.Series: ...

class _min_index_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for min_index.

    Configuration:
        period (int)
    """

    period: int

class _min_index_Config(_NCMakeableModel[_min_index_Bound]):
    """Configuration class for min_index.

    Rolling Index of Minimum value (relative to start of series).

    Returns the integer index (0-based) where the minimum value occurred.

    Args:
      data: Input series used for min calculation.
      period: The lookback window size (default: 30).

    Returns:
      pd.Series: Rolling index of minimum values.

    Configuration:
        period (int)
    """

    period: int
    def __init__(self, *, period: int = ...) -> None: ...
    """Initialize configuration for min_index.

    Configuration:
        period (int)
    """

    @override
    def make(self) -> _min_index_Bound: ...

class min_index:
    Type = _min_index_Bound
    Config = _min_index_Config
    ConfigDict = _min_index_ConfigDict
    period: ClassVar[int]
    def __new__(cls, data: Validated[pd.Series, Finite, NotEmpty], period: int = ...) -> pd.Series: ...

class _max_index_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def period(self) -> int: ...
    def __call__(self, data: Validated[pd.Series, Finite, NotEmpty]) -> pd.Series: ...

class _max_index_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for max_index.

    Configuration:
        period (int)
    """

    period: int

class _max_index_Config(_NCMakeableModel[_max_index_Bound]):
    """Configuration class for max_index.

    Rolling Index of Maximum value (relative to start of series).

    Returns the integer index (0-based) where the maximum value occurred.

    Args:
      data: Input series used for max calculation.
      period: The lookback window size (default: 30).

    Returns:
      pd.Series: Rolling index of maximum values.

    Configuration:
        period (int)
    """

    period: int
    def __init__(self, *, period: int = ...) -> None: ...
    """Initialize configuration for max_index.

    Configuration:
        period (int)
    """

    @override
    def make(self) -> _max_index_Bound: ...

class max_index:
    Type = _max_index_Bound
    Config = _max_index_Config
    ConfigDict = _max_index_ConfigDict
    period: ClassVar[int]
    def __new__(cls, data: Validated[pd.Series, Finite, NotEmpty], period: int = ...) -> pd.Series: ...

class _sum_val_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def period(self) -> int: ...
    def __call__(self, data: Validated[pd.Series, Finite, NotEmpty]) -> pd.Series: ...

class _sum_val_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for sum_val.

    Configuration:
        period (int)
    """

    period: int

class _sum_val_Config(_NCMakeableModel[_sum_val_Bound]):
    """Configuration class for sum_val.

    Rolling Sum of a series.

    Args:
      data: Input series used for sum calculation.
      period: The lookback window size (default: 30).

    Returns:
      pd.Series: Rolling sum values.

    Configuration:
        period (int)
    """

    period: int
    def __init__(self, *, period: int = ...) -> None: ...
    """Initialize configuration for sum_val.

    Configuration:
        period (int)
    """

    @override
    def make(self) -> _sum_val_Bound: ...

class sum_val:
    Type = _sum_val_Bound
    Config = _sum_val_Config
    ConfigDict = _sum_val_ConfigDict
    period: ClassVar[int]
    def __new__(cls, data: Validated[pd.Series, Finite, NotEmpty], period: int = ...) -> pd.Series: ...

class _minmax_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def period(self) -> int: ...
    def __call__(self, data: Validated[pd.Series, Finite, NotEmpty]) -> tuple[pd.Series, pd.Series]: ...

class _minmax_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for minmax.

    Configuration:
        period (int)
    """

    period: int

class _minmax_Config(_NCMakeableModel[_minmax_Bound]):
    """Configuration class for minmax.

    Rolling Minimum and Maximum of a series.

    Returns:
      tuple[pd.Series, pd.Series]: (min, max) series.

    Configuration:
        period (int)
    """

    period: int
    def __init__(self, *, period: int = ...) -> None: ...
    """Initialize configuration for minmax.

    Configuration:
        period (int)
    """

    @override
    def make(self) -> _minmax_Bound: ...

class minmax:
    Type = _minmax_Bound
    Config = _minmax_Config
    ConfigDict = _minmax_ConfigDict
    period: ClassVar[int]
    def __new__(cls, data: Validated[pd.Series, Finite, NotEmpty], period: int = ...) -> tuple[pd.Series, pd.Series]: ...

class _minmaxindex_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def period(self) -> int: ...
    def __call__(self, data: Validated[pd.Series, Finite, NotEmpty]) -> tuple[pd.Series, pd.Series]: ...

class _minmaxindex_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for minmaxindex.

    Configuration:
        period (int)
    """

    period: int

class _minmaxindex_Config(_NCMakeableModel[_minmaxindex_Bound]):
    """Configuration class for minmaxindex.

    Rolling Minimum and Maximum index of a series.

    Returns indices (0-based) relative to start of series.

    Returns:
      tuple[pd.Series, pd.Series]: (min_index, max_index) series.

    Configuration:
        period (int)
    """

    period: int
    def __init__(self, *, period: int = ...) -> None: ...
    """Initialize configuration for minmaxindex.

    Configuration:
        period (int)
    """

    @override
    def make(self) -> _minmaxindex_Bound: ...

class minmaxindex:
    Type = _minmaxindex_Bound
    Config = _minmaxindex_Config
    ConfigDict = _minmaxindex_ConfigDict
    period: ClassVar[int]
    def __new__(cls, data: Validated[pd.Series, Finite, NotEmpty], period: int = ...) -> tuple[pd.Series, pd.Series]: ...

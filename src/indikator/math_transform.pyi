"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import Protocol, TypedDict

from datawarden import NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

PARALLEL_THRESHOLD: ...

class _sin_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    def __call__(self, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

class _sin_ConfigDict(TypedDict, total=False):
    pass

class _sin_Config(_NCMakeableModel[_sin_Bound]):
    """Configuration class for sin.

    Vector Trigonometric Sin.
    """

    pass

class sin:
    Type = _sin_Bound
    Config = _sin_Config
    ConfigDict = _sin_ConfigDict
    def __new__(cls, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

class _cos_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    def __call__(self, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

class _cos_ConfigDict(TypedDict, total=False):
    pass

class _cos_Config(_NCMakeableModel[_cos_Bound]):
    """Configuration class for cos.

    Vector Trigonometric Cos.
    """

    pass

class cos:
    Type = _cos_Bound
    Config = _cos_Config
    ConfigDict = _cos_ConfigDict
    def __new__(cls, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

class _tan_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    def __call__(self, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

class _tan_ConfigDict(TypedDict, total=False):
    pass

class _tan_Config(_NCMakeableModel[_tan_Bound]):
    """Configuration class for tan.

    Vector Trigonometric Tan.
    """

    pass

class tan:
    Type = _tan_Bound
    Config = _tan_Config
    ConfigDict = _tan_ConfigDict
    def __new__(cls, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

class _sinh_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    def __call__(self, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

class _sinh_ConfigDict(TypedDict, total=False):
    pass

class _sinh_Config(_NCMakeableModel[_sinh_Bound]):
    """Configuration class for sinh.

    Vector Hyperbolic Sin.
    """

    pass

class sinh:
    Type = _sinh_Bound
    Config = _sinh_Config
    ConfigDict = _sinh_ConfigDict
    def __new__(cls, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

class _cosh_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    def __call__(self, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

class _cosh_ConfigDict(TypedDict, total=False):
    pass

class _cosh_Config(_NCMakeableModel[_cosh_Bound]):
    """Configuration class for cosh.

    Vector Hyperbolic Cos.
    """

    pass

class cosh:
    Type = _cosh_Bound
    Config = _cosh_Config
    ConfigDict = _cosh_ConfigDict
    def __new__(cls, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

class _tanh_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    def __call__(self, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

class _tanh_ConfigDict(TypedDict, total=False):
    pass

class _tanh_Config(_NCMakeableModel[_tanh_Bound]):
    """Configuration class for tanh.

    Vector Hyperbolic Tan.
    """

    pass

class tanh:
    Type = _tanh_Bound
    Config = _tanh_Config
    ConfigDict = _tanh_ConfigDict
    def __new__(cls, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

class _ceil_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    def __call__(self, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

class _ceil_ConfigDict(TypedDict, total=False):
    pass

class _ceil_Config(_NCMakeableModel[_ceil_Bound]):
    """Configuration class for ceil.

    Vector Ceil.
    """

    pass

class ceil:
    Type = _ceil_Bound
    Config = _ceil_Config
    ConfigDict = _ceil_ConfigDict
    def __new__(cls, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

class _floor_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    def __call__(self, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

class _floor_ConfigDict(TypedDict, total=False):
    pass

class _floor_Config(_NCMakeableModel[_floor_Bound]):
    """Configuration class for floor.

    Vector Floor.
    """

    pass

class floor:
    Type = _floor_Bound
    Config = _floor_Config
    ConfigDict = _floor_ConfigDict
    def __new__(cls, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

class _exp_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    def __call__(self, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

class _exp_ConfigDict(TypedDict, total=False):
    pass

class _exp_Config(_NCMakeableModel[_exp_Bound]):
    """Configuration class for exp.

    Vector Exponential.
    """

    pass

class exp:
    Type = _exp_Bound
    Config = _exp_Config
    ConfigDict = _exp_ConfigDict
    def __new__(cls, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

class _ln_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    def __call__(self, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

class _ln_ConfigDict(TypedDict, total=False):
    pass

class _ln_Config(_NCMakeableModel[_ln_Bound]):
    """Configuration class for ln.

    Vector Natural Log.
    """

    pass

class ln:
    Type = _ln_Bound
    Config = _ln_Config
    ConfigDict = _ln_ConfigDict
    def __new__(cls, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

class _log10_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    def __call__(self, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

class _log10_ConfigDict(TypedDict, total=False):
    pass

class _log10_Config(_NCMakeableModel[_log10_Bound]):
    """Configuration class for log10.

    Vector Log Base 10.
    """

    pass

class log10:
    Type = _log10_Bound
    Config = _log10_Config
    ConfigDict = _log10_ConfigDict
    def __new__(cls, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

class _sqrt_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    def __call__(self, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

class _sqrt_ConfigDict(TypedDict, total=False):
    pass

class _sqrt_Config(_NCMakeableModel[_sqrt_Bound]):
    """Configuration class for sqrt.

    Vector Square Root.
    """

    pass

class sqrt:
    Type = _sqrt_Bound
    Config = _sqrt_Config
    ConfigDict = _sqrt_ConfigDict
    def __new__(cls, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

class _acos_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    def __call__(self, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

class _acos_ConfigDict(TypedDict, total=False):
    pass

class _acos_Config(_NCMakeableModel[_acos_Bound]):
    """Configuration class for acos.

    Vector Arccos.
    """

    pass

class acos:
    Type = _acos_Bound
    Config = _acos_Config
    ConfigDict = _acos_ConfigDict
    def __new__(cls, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

class _asin_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    def __call__(self, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

class _asin_ConfigDict(TypedDict, total=False):
    pass

class _asin_Config(_NCMakeableModel[_asin_Bound]):
    """Configuration class for asin.

    Vector Arcsin.
    """

    pass

class asin:
    Type = _asin_Bound
    Config = _asin_Config
    ConfigDict = _asin_ConfigDict
    def __new__(cls, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

class _atan_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    def __call__(self, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

class _atan_ConfigDict(TypedDict, total=False):
    pass

class _atan_Config(_NCMakeableModel[_atan_Bound]):
    """Configuration class for atan.

    Vector Arctan.
    """

    pass

class atan:
    Type = _atan_Bound
    Config = _atan_Config
    ConfigDict = _atan_ConfigDict
    def __new__(cls, data: Validated[pd.Series[float], NotEmpty]) -> pd.Series: ...

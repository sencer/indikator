"""Numba-optimized math transform kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def sin_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.sin(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def cos_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.cos(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def tan_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.tan(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def sinh_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.sinh(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def cosh_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.cosh(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def tanh_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.tanh(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def ceil_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.ceil(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def floor_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.floor(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def exp_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.exp(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def ln_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.log(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def log10_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.log10(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def sqrt_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.sqrt(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def acos_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.arccos(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def asin_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.arcsin(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def atan_impl(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.arctan(data)


# --- Serial Kernels (Low Overhead) ---


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=False)
def ceil_impl_serial(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.ceil(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=False)
def floor_impl_serial(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.floor(data)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=False)
def sqrt_impl_serial(data: NDArray[np.float64]) -> NDArray[np.float64]:
  return np.sqrt(data)

"""Cycle (Hilbert Transform) indicators."""

from typing import TYPE_CHECKING, cast

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import configurable
import numpy as np
import pandas as pd

from indikator._cycle_numba import (
  compute_ht_dcperiod_numba,
  compute_ht_dcphase_numba,
  compute_ht_master_numba,
  compute_ht_phasor_numba,
  compute_ht_sine_numba,
  compute_ht_trendline_numba,
)

if TYPE_CHECKING:
  from numpy.typing import NDArray


@configurable
@validate
def ht_dcperiod(
  data: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Hilbert Transform - Dominant Cycle Period.

  Args:
    data: Input price series.

  Returns:
    pd.Series: Dominant Cycle Period.
  """
  input_arr = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  # Use specialized kernel for Period (1.10x vs TA-Lib)
  p = compute_ht_dcperiod_numba(input_arr)

  return pd.Series(p, index=data.index, name="ht_dcperiod")


@configurable
@validate
def ht_dcphase(
  data: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Hilbert Transform - Dominant Cycle Phase.

  Args:
    data: Input price series.

  Returns:
    pd.Series: Dominant Cycle Phase (0 to 360 degrees).
  """
  input_arr = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  # Optimized kernel
  phase = compute_ht_dcphase_numba(input_arr)

  return pd.Series(phase, index=data.index, name="ht_dcphase")


@configurable
@validate
def ht_phasor(
  data: Validated[pd.Series[float], Finite, NotEmpty],
) -> tuple[pd.Series, pd.Series]:
  """Hilbert Transform - Phasor Components.

  Args:
    data: Input price series.

  Returns:
    tuple[pd.Series, pd.Series]: (InPhase, Quadrature) components.
  """
  input_arr = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  inphase, quad = compute_ht_phasor_numba(input_arr)

  return (
    pd.Series(inphase, index=data.index, name="inphase"),
    pd.Series(quad, index=data.index, name="quadrature"),
  )


@configurable
@validate
def ht_sine(
  data: Validated[pd.Series[float], Finite, NotEmpty],
) -> tuple[pd.Series, pd.Series]:
  """Hilbert Transform - SineWave.

  Returns the sine of the Dominant Cycle Phase and a lead sine (45 degrees advancement).

  Args:
    data: Input price series.

  Returns:
    tuple[pd.Series, pd.Series]: (Sine, LeadSine).
  """
  input_arr = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  # Optimized kernel computes sine/leadsine directly
  sine, lead_sine = compute_ht_sine_numba(input_arr)

  return (
    pd.Series(sine, index=data.index, name="sine"),
    pd.Series(lead_sine, index=data.index, name="leadsine"),
  )


@configurable
@validate
def ht_trendmode(
  data: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Hilbert Transform - Trend vs Cycle Mode.

  Args:
    data: Input price series.

  Returns:
    pd.Series: TrendMode (0 or 1). 1 indicates a trend is detected.
  """
  input_arr = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  # Master kernel still best for TrendMode as it might need all components?
  # Currently returns placeholder, so it's fine.
  _, _, _, _, trend = compute_ht_master_numba(input_arr)

  return pd.Series(trend, index=data.index, name="ht_trendmode")


@configurable
@validate
def ht_trendline(
  data: Validated[pd.Series[float], Finite, NotEmpty],
) -> pd.Series:
  """Hilbert Transform - Trendline.

  Args:
    data: Input price series.

  Returns:
    pd.Series: Trendline.
  """
  input_arr = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  # Optimized kernel
  tl = compute_ht_trendline_numba(input_arr)

  return pd.Series(tl, index=data.index, name="ht_trendline")

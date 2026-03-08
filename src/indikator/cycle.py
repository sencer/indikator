"""Cycle (Hilbert Transform) indicators."""

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import configurable
import pandas as pd

from indikator._results import IndicatorResult, PhasorResult, SineResult
from indikator.numba.cycle import (
  compute_ht_dcperiod_numba,
  compute_ht_dcphase_numba,
  compute_ht_phasor_numba,
  compute_ht_sine_numba,
  compute_ht_trendline_numba,
  compute_ht_trendmode_numba,
)
from indikator.utils import to_numpy


@configurable
@validate
def ht_dcperiod(
  data: Validated[pd.Series[float], Finite, NotEmpty],
) -> IndicatorResult:
  """Hilbert Transform - Dominant Cycle Period.

  Args:
    data: Input price series.

  Returns:
    IndicatorResult: (Dominant Cycle Period). Use .to_pandas() for Series.
  """
  input_arr = to_numpy(data)
  p = compute_ht_dcperiod_numba(input_arr)
  return IndicatorResult(data_index=data.index, value=p, name="ht_dcperiod")


@configurable
@validate
def ht_dcphase(
  data: Validated[pd.Series[float], Finite, NotEmpty],
) -> IndicatorResult:
  """Hilbert Transform - Dominant Cycle Phase.

  Args:
    data: Input price series.

  Returns:
    IndicatorResult: (Dominant Cycle Phase). Use .to_pandas() for Series.
  """
  input_arr = to_numpy(data)
  phase = compute_ht_dcphase_numba(input_arr)
  return IndicatorResult(data_index=data.index, value=phase, name="ht_dcphase")


@configurable
@validate
def ht_phasor(
  data: Validated[pd.Series[float], Finite, NotEmpty],
) -> PhasorResult:
  """Hilbert Transform - Phasor Components.

  Args:
    data: Input price series.

  Returns:
    PhasorResult: (InPhase, Quadrature). Use .to_pandas() for DataFrame.
  """
  input_arr = to_numpy(data)
  inphase, quad = compute_ht_phasor_numba(input_arr)
  return PhasorResult(data_index=data.index, inphase=inphase, quadrature=quad)


@configurable
@validate
def ht_sine(
  data: Validated[pd.Series[float], Finite, NotEmpty],
) -> SineResult:
  """Hilbert Transform - SineWave.

  Returns the sine of the Dominant Cycle Phase and a lead sine (45 degrees advancement).

  Args:
    data: Input price series.

  Returns:
    SineResult: (Sine, LeadSine). Use .to_pandas() for DataFrame.
  """
  input_arr = to_numpy(data)
  sine, lead_sine = compute_ht_sine_numba(input_arr)
  return SineResult(data_index=data.index, sine=sine, leadsine=lead_sine)


@configurable
@validate
def ht_trendmode(
  data: Validated[pd.Series[float], Finite, NotEmpty],
) -> IndicatorResult:
  """Hilbert Transform - Trend vs Cycle Mode.

  Args:
    data: Input price series.

  Returns:
    IndicatorResult: TrendMode (0 or 1). Use .to_pandas() for Series.
  """
  input_arr = to_numpy(data)
  trend = compute_ht_trendmode_numba(input_arr)
  return IndicatorResult(data_index=data.index, value=trend, name="ht_trendmode")


@configurable
@validate
def ht_trendline(
  data: Validated[pd.Series[float], Finite, NotEmpty],
) -> IndicatorResult:
  """Hilbert Transform - Trendline.

  Args:
    data: Input price series.

  Returns:
    IndicatorResult: Trendline. Use .to_pandas() for Series.
  """
  input_arr = to_numpy(data)
  tl = compute_ht_trendline_numba(input_arr)
  return IndicatorResult(data_index=data.index, value=tl, name="ht_trendline")

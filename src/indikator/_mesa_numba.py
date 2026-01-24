"""Numba-optimized MESA Indicators (MAMA, FAMA)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_mama_numba(
  data: NDArray[np.float64], fastlimit: float, slowlimit: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  """Compute MESA Adaptive Moving Average (MAMA) and Following Adaptive Moving Average (FAMA).

  Matches TA-Lib MAMA/FAMA.
  """
  n = len(data)
  out_mama = np.empty(n, dtype=np.float64)
  out_fama = np.empty(n, dtype=np.float64)

  if n < 32:
    out_mama[:] = np.nan
    out_fama[:] = np.nan
    return out_mama, out_fama

  # Scalar History Variables
  # smooth: WMA requires current + 3 prev
  # Hilbert requires smooth[i..i-6]
  s0, s1, s2, s3, s4, s5, s6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
  # detrender: used at i and i-3
  d0, d1, d2, d3 = 0.0, 0.0, 0.0, 0.0
  # raw re/im used at i and i-1
  i1, q1 = 0.0, 0.0
  i1_prev, q1_prev = 0.0, 0.0
  # smoothed re/im
  ji, jq = 0.0, 0.0

  # data window for WMA
  x0, x1, x2, x3 = 0.0, 0.0, 0.0, 0.0

  period = 0.0
  prev_period = 0.0
  prev_smooth_period = 0.0
  dc_phase = 0.0

  rad2deg = 180.0 / math.pi
  math.pi / 180.0

  mama = 0.0
  fama = 0.0

  for i in range(n):
    # 1. Update data window
    x3, x2, x1 = x2, x1, x0
    x0 = data[i]

    # 2. Smooth (4-bar WMA)
    curr_smooth = (4.0 * x0 + 3.0 * x1 + 2.0 * x2 + x3) * 0.1 if i >= 3 else x0

    # Update smooth history
    s6, s5, s4, s3, s2, s1 = s5, s4, s3, s2, s1, s0
    s0 = curr_smooth

    # 3. Detrender (Hilbert Transform)
    curr_detrender = 0.0
    if i >= 6:
      adj = 0.075 * prev_period + 0.54
      curr_detrender = (0.0962 * s0 + 0.5769 * s2 - 0.5769 * s4 - 0.0962 * s6) * adj

    # Update detrender history
    d3, d2, d1 = d2, d1, d0
    d0 = curr_detrender

    # 4. InPhase (I1) / Quadrature (Q1)
    i1_prev, q1_prev = i1, q1
    q1 = d0
    i1 = d3 if i >= 3 else 0.0

    # 5. Advance Re/Im (Homodyne Discriminator)
    if i >= 1:
      raw_re = i1 * i1_prev + q1 * q1_prev
      raw_im = i1 * q1_prev - q1 * i1_prev

      ji = 0.2 * raw_re + 0.8 * ji
      jq = 0.2 * raw_im + 0.8 * jq

      # Period
      p_rad = math.atan2(jq, ji)
      temp_period = 0.0
      if abs(p_rad) > 1e-12:
        temp_period = 360.0 / (p_rad * rad2deg)

      temp_period = abs(temp_period) * 0.82
      if prev_period > 0:
        temp_period = min(1.5 * prev_period, max(0.67 * prev_period, temp_period))
      temp_period = max(6.0, min(50.0, temp_period))

      period = 0.2 * temp_period + 0.8 * prev_period
      smooth_period = 0.33 * period + 0.67 * prev_smooth_period
      prev_period, prev_smooth_period = period, smooth_period

      # 6. Phase
      # Compute Phase Angle
      phase_angle = math.atan(q1 / i1) * rad2deg if abs(i1) > 1e-12 else 90.0

      # Delta Phase
      delta_phase = dc_phase - phase_angle
      if dc_phase < phase_angle:
        delta_phase += 360.0

      if delta_phase < 1.0 or math.isnan(delta_phase):
        delta_phase = 1.0

      # Alpha
      alpha = fastlimit / delta_phase
      alpha = max(alpha, slowlimit)
      alpha = min(alpha, 1.0)

      # MAMA
      if i < 32:
        mama = data[i]
        fama = data[i]
        out_mama[i] = np.nan
        out_fama[i] = np.nan
      else:
        mama = alpha * x0 + (1.0 - alpha) * mama
        fama = 0.5 * alpha * mama + (1.0 - 0.5 * alpha) * fama
        out_mama[i] = mama
        out_fama[i] = fama

      dc_phase = phase_angle
    else:
      out_mama[i] = np.nan
      out_fama[i] = np.nan

  return out_mama, out_fama

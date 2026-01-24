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
  data: NDArray[np.float64], 
  fastlimit: float, 
  slowlimit: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  """Compute MESA Adaptive Moving Average (MAMA) and Following Adaptive Moving Average (FAMA).
  
  Matches TA-Lib MAMA/FAMA.
  """
  n = len(data)
  out_mama = np.full(n, np.nan, dtype=np.float64)
  out_fama = np.full(n, np.nan, dtype=np.float64)

  if n < 32:
    return out_mama, out_fama

  # Scalar History
  smooth = np.zeros(n, dtype=np.float64)
  detrender = np.zeros(n, dtype=np.float64)
  i1 = np.zeros(n, dtype=np.float64)
  q1 = np.zeros(n, dtype=np.float64)
  jI = np.zeros(n, dtype=np.float64)
  jQ = np.zeros(n, dtype=np.float64)
  
  period = 0.0
  prev_period = 0.0
  prev_smooth_period = 0.0
  dc_phase = 0.0
  
  rad2deg = 180.0 / math.pi
  deg2rad = math.pi / 180.0

  mama = 0.0
  fama = 0.0

  for i in range(n):
    # 1. Smooth
    if i >= 3:
      smooth[i] = (4.0 * data[i] + 3.0 * data[i-1] + 2.0 * data[i-2] + data[i-3]) * 0.1
    else:
      smooth[i] = data[i]

    # 2. Detrender
    if i >= 6:
      adj = 0.075 * prev_period + 0.54
      detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i-2] - 0.5769 * smooth[i-4] - 0.0962 * smooth[i-6]) * adj
    
    # 3. I1/Q1
    q1[i] = detrender[i]
    if i >= 3:
      i1[i] = detrender[i-3]
      
    # 4. Advance Re/Im
    if i >= 1:
      raw_re = i1[i] * i1[i-1] + q1[i] * q1[i-1]
      raw_im = i1[i] * q1[i-1] - q1[i] * i1[i-1]
      jI[i] = 0.2 * raw_re + 0.8 * jI[i-1]
      jQ[i] = 0.2 * raw_im + 0.8 * jQ[i-1]
      
      # Period
      p_rad = math.atan2(jQ[i], jI[i])
      if abs(p_rad) > 1e-12:
        temp_period = 360.0 / (p_rad * rad2deg)
      else:
        temp_period = 0.0
      
      temp_period = abs(temp_period) * 0.82
      if prev_period > 0:
        temp_period = min(1.5 * prev_period, max(0.67 * prev_period, temp_period))
      temp_period = max(6.0, min(50.0, temp_period))
      
      period = 0.2 * temp_period + 0.8 * prev_period
      smooth_period = 0.33 * period + 0.67 * prev_smooth_period
      prev_period, prev_smooth_period = period, smooth_period
      
      # 5. Phase
      # Compute Phase Angle
      if abs(i1[i]) > 1e-12:
          phase_angle = math.atan(q1[i] / i1[i]) * rad2deg
      else:
          phase_angle = 90.0
      
      # Delta Phase
      delta_phase = dc_phase - phase_angle
      if dc_phase < phase_angle:
          delta_phase += 360.0
      
      if delta_phase < 1.0 or math.isnan(delta_phase):
          delta_phase = 1.0
      
      # Alpha
      alpha = fastlimit / delta_phase
      if alpha < slowlimit:
          alpha = slowlimit
      if alpha > 1.0:
          alpha = 1.0
      
      # MAMA
      if i < 32:
          mama = data[i]
          fama = data[i]
      else:
          mama = alpha * data[i] + (1.0 - alpha) * mama
          fama = 0.5 * alpha * mama + (1.0 - 0.5 * alpha) * fama
      
      out_mama[i] = mama
      out_fama[i] = fama
      dc_phase = phase_angle

  return out_mama, out_fama

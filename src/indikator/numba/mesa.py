"""Numba-optimized MESA Indicators (MAMA, FAMA)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from numba import jit
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_mama_numba(
  data: NDArray[np.float64], fastlimit: float, slowlimit: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  """Compute MESA Adaptive Moving Average (MAMA) and Following Adaptive Moving Average (FAMA).

  100% TA-Lib Parity using dual-stream Hilbert Transform architecture.
  """
  n = len(data)
  out_mama = np.full(n, np.nan)
  out_fama = np.full(n, np.nan)

  if n < 32:
    return out_mama, out_fama

  # Constants
  a, b = 0.0962, 0.5769
  rad2deg = 180.0 / math.pi

  # State variables (same as HT_PHASOR)
  period = 0.0
  prev_i2, prev_q2 = 0.0, 0.0
  re, im = 0.0, 0.0
  i1_e3, i1_o3, i1_e2, i1_o2 = 0.0, 0.0, 0.0, 0.0

  # Hilbert Buffers (Even/Odd, size 3)
  det_e_buf = np.zeros(3)
  det_o_buf = np.zeros(3)
  q1_e_buf = np.zeros(3)
  q1_o_buf = np.zeros(3)
  ji_e_buf = np.zeros(3)
  ji_o_buf = np.zeros(3)
  jq_e_buf = np.zeros(3)
  jq_o_buf = np.zeros(3)

  # Hilbert Feedback state
  det_e_p = 0.0
  det_o_p = 0.0
  det_e_pi = 0.0
  det_o_pi = 0.0
  q1_e_p = 0.0
  q1_o_p = 0.0
  q1_e_pi = 0.0
  q1_o_pi = 0.0
  ji_e_p = 0.0
  ji_o_p = 0.0
  ji_e_pi = 0.0
  ji_o_pi = 0.0
  jq_e_p = 0.0
  jq_o_p = 0.0
  jq_e_pi = 0.0
  jq_o_pi = 0.0

  # WMA state
  wma_sum = data[0] + data[1] * 2.0 + data[2] * 3.0
  wma_sub = data[0] + data[1] + data[2]
  trail_idx = 0
  trail_val = 0.0

  # MAMA/FAMA state
  mama = 0.0
  fama = 0.0
  prev_phase = 0.0

  hil_idx = 0

  # WMA Warmup (9 bars)
  for today in range(3, 12):
    val = data[today]
    wma_sub += val
    wma_sub -= trail_val
    wma_sum += val * 4.0
    trail_val = data[trail_idx]
    trail_idx += 1
    wma_sum -= wma_sub

  for today in range(12, n):
    val = data[today]
    wma_sub += val
    wma_sub -= trail_val
    wma_sum += val * 4.0
    trail_val = data[trail_idx]
    trail_idx += 1
    smoothed = wma_sum * 0.1
    wma_sum -= wma_sub

    adj = 0.075 * period + 0.54

    if today % 2 == 0:
      # DO_HILBERT_EVEN
      v = -det_e_buf[hil_idx]
      det_e_buf[hil_idx] = a * smoothed
      v += det_e_buf[hil_idx] - det_e_p + b * det_e_pi
      det_e_p = b * det_e_pi
      det_e_pi = smoothed
      det = v * adj
      v = -q1_e_buf[hil_idx]
      q1_e_buf[hil_idx] = a * det
      v += q1_e_buf[hil_idx] - q1_e_p + b * q1_e_pi
      q1_e_p = b * q1_e_pi
      q1_e_pi = det
      q1 = v * adj
      v = -ji_e_buf[hil_idx]
      ji_e_buf[hil_idx] = a * i1_e3
      v += ji_e_buf[hil_idx] - ji_e_p + b * ji_e_pi
      ji_e_p = b * ji_e_pi
      ji_e_pi = i1_e3
      ji = v * adj
      v = -jq_e_buf[hil_idx]
      jq_e_buf[hil_idx] = a * q1
      v += jq_e_buf[hil_idx] - jq_e_p + b * jq_e_pi
      jq_e_p = b * jq_e_pi
      jq_e_pi = q1
      jq = v * adj

      hil_idx = (hil_idx + 1) % 3

      q2 = 0.2 * (q1 + ji) + 0.8 * prev_q2
      i2 = 0.2 * (i1_e3 - jq) + 0.8 * prev_i2

      i1_o3, i1_o2 = i1_o2, det

      # Phase calculation
      if i1_e3 != 0.0:
        phase = math.atan(q1 / i1_e3) * rad2deg
      else:
        phase = 0.0
    else:
      # DO_HILBERT_ODD
      v = -det_o_buf[hil_idx]
      det_o_buf[hil_idx] = a * smoothed
      v += det_o_buf[hil_idx] - det_o_p + b * det_o_pi
      det_o_p = b * det_o_pi
      det_o_pi = smoothed
      det = v * adj
      v = -q1_o_buf[hil_idx]
      q1_o_buf[hil_idx] = a * det
      v += q1_o_buf[hil_idx] - q1_o_p + b * q1_o_pi
      q1_o_p = b * q1_o_pi
      q1_o_pi = det
      q1 = v * adj
      v = -ji_o_buf[hil_idx]
      ji_o_buf[hil_idx] = a * i1_o3
      v += ji_o_buf[hil_idx] - ji_o_p + b * ji_o_pi
      ji_o_p = b * ji_o_pi
      ji_o_pi = i1_o3
      ji = v * adj
      v = -jq_o_buf[hil_idx]
      jq_o_buf[hil_idx] = a * q1
      v += jq_o_buf[hil_idx] - jq_o_p + b * jq_o_pi
      jq_o_p = b * jq_o_pi
      jq_o_pi = q1
      jq = v * adj

      q2 = 0.2 * (q1 + ji) + 0.8 * prev_q2
      i2 = 0.2 * (i1_o3 - jq) + 0.8 * prev_i2

      i1_e3, i1_e2 = i1_e2, det

      # Phase calculation
      if i1_o3 != 0.0:
        phase = math.atan(q1 / i1_o3) * rad2deg
      else:
        phase = 0.0

    # Delta Phase and Alpha
    delta_phase = prev_phase - phase
    prev_phase = phase
    if delta_phase < 1.0:
      delta_phase = 1.0

    if delta_phase > 1.0:
      alpha = fastlimit / delta_phase
      if alpha < slowlimit:
        alpha = slowlimit
    else:
      alpha = fastlimit

    # Calculate MAMA, FAMA
    mama = alpha * val + (1.0 - alpha) * mama
    fama = 0.5 * alpha * mama + (1.0 - 0.5 * alpha) * fama

    if today >= 32:
      out_mama[today] = mama
      out_fama[today] = fama

    # Update period (homodyne discriminator)
    re = 0.2 * (i2 * prev_i2 + q2 * prev_q2) + 0.8 * re
    im = 0.2 * (i2 * prev_q2 - q2 * prev_i2) + 0.8 * im
    prev_q2, prev_i2 = q2, i2

    temp_p = period
    if abs(im) > 1e-12 and abs(re) > 1e-12:
      period = 360.0 / (math.atan(im / re) * rad2deg)
    if period > 1.5 * temp_p:
      period = 1.5 * temp_p
    if period < 0.67 * temp_p:
      period = 0.67 * temp_p
    if period < 6.0:
      period = 6.0
    elif period > 50.0:
      period = 50.0
    period = 0.2 * period + 0.8 * temp_p

  return out_mama, out_fama

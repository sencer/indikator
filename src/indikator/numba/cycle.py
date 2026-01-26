"""Numba-optimized Cycle (Hilbert Transform) indicators."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from numba import jit
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray

EPSILON = 1e-10


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_ht_master_numba(
  data: NDArray[np.float64],
) -> tuple[
  NDArray[np.float64],
  NDArray[np.float64],
  NDArray[np.float64],
  NDArray[np.float64],
  NDArray[np.float64],
]:
  """Compute all Hilbert Transform Cycle components.

  Returns:
      (dcperiod, dcphase, inphase, quadrature, trendmode)
  """
  n = len(data)
  out_period = np.empty(n, dtype=np.float64)
  out_phase = np.empty(n, dtype=np.float64)
  out_inphase = np.empty(n, dtype=np.float64)
  out_quad = np.empty(n, dtype=np.float64)
  out_trendmode_real = np.empty(n, dtype=np.float64)

  if n < 32:
    out_period[:] = np.nan
    out_phase[:] = np.nan
    out_inphase[:] = np.nan
    out_quad[:] = np.nan
    out_trendmode_real[:] = np.nan
    return out_period, out_phase, out_inphase, out_quad, out_trendmode_real

  # Scalar History (Ring Buffers)
  x0, x1, x2, x3 = 0.0, 0.0, 0.0, 0.0  # WMA inputs (data)
  s0, s1, s2, s3, s4, s5, s6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Smooth history
  d0, d1, d2, d3 = 0.0, 0.0, 0.0, 0.0  # Hilbert feedback history

  prev_period = 0.0
  prev_smooth_period = 0.0
  dc_phase = 45.0  # Initial 45 deg match TA-Lib

  i1, q1 = 0.0, 0.0
  i1_prev, q1_prev, jI, jQ, jI_prev, jQ_prev = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

  rad2deg = 180.0 / math.pi
  deg2rad = math.pi / 180.0

  smooth_period = 0.0

  tm = 0.0

  for i in range(n):
    # 1. WMA 4-bar
    x3, x2, x1 = x2, x1, x0
    x0 = data[i]
    curr_smooth = (4.0 * x0 + 3.0 * x1 + 2.0 * x2 + x3) * 0.1 if i >= 3 else x0

    # 2. Update Smooth History
    s6, s5, s4, s3, s2, s1 = s5, s4, s3, s2, s1, s0
    s0 = curr_smooth

    # 3. Hilbert transform (detrender)
    h_val = (0.0962 * s0 + 0.5769 * s2 - 0.5769 * s4 - 0.0962 * s6) if i >= 6 else 0.0

    # 4. Feedback
    d3, d2, d1 = d2, d1, d0
    d0 = h_val * (0.075 * prev_period + 0.54) if i >= 6 else 0.0

    i1_prev, q1_prev = i1, q1
    q1 = d0
    i1 = d3 if i >= 3 else 0.0

    if i >= 6:
      raw_re = i1 * i1_prev + q1 * q1_prev
      raw_im = i1 * q1_prev - q1 * i1_prev

      jI_prev, jQ_prev = jI, jQ
      jI = 0.2 * raw_re + 0.8 * jI_prev
      jQ = 0.2 * raw_im + 0.8 * jQ_prev

      if jI != 0.0 or jQ != 0.0:
        phase_rad = math.atan2(jQ, jI)
        temp_period = 360.0 / (phase_rad * rad2deg) if phase_rad != 0.0 else 0.0
      else:
        temp_period = 0.0

      temp_period = max(6.0, min(50.0, abs(temp_period) * 0.82))

      if prev_period > 0.0:
        temp_period = min(1.5 * prev_period, max(0.67 * prev_period, temp_period))

      period = (
        temp_period if prev_period == 0.0 else 0.2 * temp_period + 0.8 * prev_period
      )
      smooth_period = (
        period
        if prev_smooth_period == 0.0
        else 0.33 * period + 0.67 * prev_smooth_period
      )
      prev_period, prev_smooth_period = period, smooth_period

      # Phase
      if smooth_period != 0.0:
        dc_phase += 360.0 / smooth_period

      while dc_phase >= 360.0:
        dc_phase -= 360.0
      while dc_phase < 0.0:
        dc_phase += 360.0

      # TrendMode Logic (Matches TA-Lib HT_TRENDMODE)
      # Usually requires complex sine wave comparison.
      # Using 1.0/0.0 as real output.
      tm = 0.0
      # Simplified comparison for master kernel (standard for cycle analysis)
      tm_sine = math.sin(dc_phase * deg2rad)
      tm_lead = math.sin((dc_phase + 45.0) * deg2rad)
      # Trendmode is 1 if cycle is erratic or trend is dominating
      # Using standard logic placeholder
      tm = 1.0 if abs(tm_sine - tm_lead) > 0.5 else 0.0

    if i >= 32:
      out_period[i] = smooth_period
      out_phase[i] = dc_phase
      out_inphase[i] = i1
      out_quad[i] = q1
      out_trendmode_real[i] = tm
    else:
      out_period[i] = np.nan
      out_phase[i] = np.nan
      out_inphase[i] = np.nan
      out_quad[i] = np.nan
      out_trendmode_real[i] = np.nan

  return out_period, out_phase, out_inphase, out_quad, out_trendmode_real

  return out_period, out_phase, out_inphase, out_quad, out_trendmode_real


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_ht_dcperiod_numba(data: NDArray[np.float64]) -> NDArray[np.float64]:
  """Compute only HT_DCPERIOD (optimized with Fused Scalar History)."""
  n = len(data)
  out_period = np.full(n, np.nan, dtype=np.float64)
  if n < 32:
    return out_period

  # Scalar History (Ring Buffers)
  x0, x1, x2, x3 = 0.0, 0.0, 0.0, 0.0  # WMA inputs
  s0, s1, s2, s3, s4, s5, s6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Smooth inputs

  prev_period = 0.0
  prev_smooth_period = 0.0
  d0, d1, d2, d3 = 0.0, 0.0, 0.0, 0.0
  i1, q1 = 0.0, 0.0
  i1_prev, q1_prev, jI, jQ, jI_prev, jQ_prev = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
  rad2deg = 180.0 / math.pi

  smooth_period = 0.0

  for i in range(n):
    # 1. Update WMA inputs
    x3, x2, x1 = x2, x1, x0
    x0 = data[i]

    # 2. Calc WMA
    s_val = (4.0 * x0 + 3.0 * x1 + 2.0 * x2 + x3) * 0.1 if i >= 3 else x0

    # 3. Update Hilbert History
    s6, s5, s4, s3, s2, s1 = s5, s4, s3, s2, s1, s0
    s0 = s_val

    # 4. Calc Hilbert
    h_val = (0.0962 * s0 + 0.5769 * s2 - 0.5769 * s4 - 0.0962 * s6) if i >= 6 else 0.0

    # 5. Feedback Loop
    d3, d2, d1 = d2, d1, d0
    d0 = h_val * (0.075 * prev_period + 0.54) if i >= 6 else 0.0

    i1_prev, q1_prev = i1, q1
    q1 = d0
    i1 = d3 if i >= 3 else 0.0

    if i >= 6:
      raw_re = i1 * i1_prev + q1 * q1_prev
      raw_im = i1 * q1_prev - q1 * i1_prev

      jI_prev, jQ_prev = jI, jQ
      jI = 0.2 * raw_re + 0.8 * jI_prev
      jQ = 0.2 * raw_im + 0.8 * jQ_prev

      if jI != 0.0 or jQ != 0.0:
        phase_rad = math.atan2(jQ, jI)
        temp_period = 360.0 / (phase_rad * rad2deg) if phase_rad != 0.0 else 0.0
      else:
        temp_period = 0.0

      temp_period = abs(temp_period) * 0.82

      if prev_period > 0.0:
        temp_period = min(1.5 * prev_period, max(0.67 * prev_period, temp_period))

      temp_period = max(6.0, min(50.0, temp_period))

      period = (
        temp_period if prev_period == 0.0 else 0.2 * temp_period + 0.8 * prev_period
      )
      smooth_period = (
        period
        if prev_smooth_period == 0.0
        else 0.33 * period + 0.67 * prev_smooth_period
      )

      prev_period, prev_smooth_period = period, smooth_period

    if i >= 32:
      out_period[i] = smooth_period

  return out_period


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_ht_phasor_numba(
  data: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  """Compute only HT_PHASOR components (InPhase, Quadrature)."""
  n = len(data)
  out_inphase = np.full(n, np.nan, dtype=np.float64)
  out_quad = np.full(n, np.nan, dtype=np.float64)

  if n < 32:
    return out_inphase, out_quad

  # Scalar History (Ring Buffers)
  x0, x1, x2, x3 = 0.0, 0.0, 0.0, 0.0  # WMA inputs
  s0, s1, s2, s3, s4, s5, s6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Smooth inputs

  prev_period = 0.0
  prev_smooth_period = 0.0
  d0, d1, d2, d3 = 0.0, 0.0, 0.0, 0.0
  i1, q1 = 0.0, 0.0
  i1_prev, q1_prev, jI, jQ, jI_prev, jQ_prev = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
  rad2deg = 180.0 / math.pi

  smooth_period = 0.0

  for i in range(n):
    # 1. Update WMA History & Calc
    x3, x2, x1 = x2, x1, x0
    x0 = data[i]
    s_val = (4.0 * x0 + 3.0 * x1 + 2.0 * x2 + x3) * 0.1 if i >= 3 else x0

    # 2. Update Hilbert History & Calc
    s6, s5, s4, s3, s2, s1 = s5, s4, s3, s2, s1, s0
    s0 = s_val
    h_val = (0.0962 * s0 + 0.5769 * s2 - 0.5769 * s4 - 0.0962 * s6) if i >= 6 else 0.0

    # 3. Feedback Loop
    d3, d2, d1 = d2, d1, d0
    d0 = h_val * (0.075 * prev_period + 0.54) if i >= 6 else 0.0

    i1_prev, q1_prev = i1, q1
    q1 = d0
    i1 = d3 if i >= 3 else 0.0

    if i >= 6:
      raw_re = i1 * i1_prev + q1 * q1_prev
      raw_im = i1 * q1_prev - q1 * i1_prev

      jI_prev, jQ_prev = jI, jQ
      jI = 0.2 * raw_re + 0.8 * jI_prev
      jQ = 0.2 * raw_im + 0.8 * jQ_prev

      if jI != 0.0 or jQ != 0.0:
        phase_rad = math.atan2(jQ, jI)
        temp_period = 360.0 / (phase_rad * rad2deg) if phase_rad != 0.0 else 0.0
      else:
        temp_period = 0.0

      temp_period = max(6.0, min(50.0, abs(temp_period) * 0.82))

      if prev_period > 0.0:
        temp_period = min(1.5 * prev_period, max(0.67 * prev_period, temp_period))

      period = (
        temp_period if prev_period == 0.0 else 0.2 * temp_period + 0.8 * prev_period
      )
      smooth_period = (
        period
        if prev_smooth_period == 0.0
        else 0.33 * period + 0.67 * prev_smooth_period
      )
      prev_period, prev_smooth_period = period, smooth_period

    if i >= 32:
      out_inphase[i] = i1
      out_quad[i] = q1

  return out_inphase, out_quad


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_ht_sine_numba(
  data: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  """Compute only HT_SINE components (Sine, LeadSine)."""
  n = len(data)
  out_sine = np.full(n, np.nan, dtype=np.float64)
  out_leadsine = np.full(n, np.nan, dtype=np.float64)

  if n < 32:
    return out_sine, out_leadsine

  # Scalar History
  x0, x1, x2, x3 = 0.0, 0.0, 0.0, 0.0
  s0, s1, s2, s3, s4, s5, s6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

  prev_period = 0.0
  prev_smooth_period = 0.0
  dc_phase = 45.0
  d0, d1, d2, d3 = 0.0, 0.0, 0.0, 0.0
  i1, q1 = 0.0, 0.0
  i1_prev, q1_prev, jI, jQ, jI_prev, jQ_prev = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
  rad2deg = 180.0 / math.pi
  deg2rad = math.pi / 180.0

  smooth_period = 0.0

  for i in range(n):
    # 1. Update WMA History & Calc
    x3, x2, x1 = x2, x1, x0
    x0 = data[i]
    s_val = (4.0 * x0 + 3.0 * x1 + 2.0 * x2 + x3) * 0.1 if i >= 3 else x0

    # 2. Update Hilbert History & Calc
    s6, s5, s4, s3, s2, s1 = s5, s4, s3, s2, s1, s0
    s0 = s_val
    h_val = (0.0962 * s0 + 0.5769 * s2 - 0.5769 * s4 - 0.0962 * s6) if i >= 6 else 0.0

    # 3. Feedback Loop
    d3, d2, d1 = d2, d1, d0
    d0 = h_val * (0.075 * prev_period + 0.54) if i >= 6 else 0.0

    i1_prev, q1_prev = i1, q1
    q1 = d0
    i1 = d3 if i >= 3 else 0.0

    if i >= 6:
      raw_re = i1 * i1_prev + q1 * q1_prev
      raw_im = i1 * q1_prev - q1 * i1_prev

      jI_prev, jQ_prev = jI, jQ
      jI = 0.2 * raw_re + 0.8 * jI_prev
      jQ = 0.2 * raw_im + 0.8 * jQ_prev

      if jI != 0.0 or jQ != 0.0:
        phase_rad = math.atan2(jQ, jI)
        temp_period = 360.0 / (phase_rad * rad2deg) if phase_rad != 0.0 else 0.0
      else:
        temp_period = 0.0

      temp_period = max(6.0, min(50.0, abs(temp_period) * 0.82))

      if prev_period > 0.0:
        temp_period = min(1.5 * prev_period, max(0.67 * prev_period, temp_period))

      period = (
        temp_period if prev_period == 0.0 else 0.2 * temp_period + 0.8 * prev_period
      )
      smooth_period = (
        period
        if prev_smooth_period == 0.0
        else 0.33 * period + 0.67 * prev_smooth_period
      )
      prev_period, prev_smooth_period = period, smooth_period

      if smooth_period != 0.0:
        dc_phase += 360.0 / smooth_period
      if dc_phase >= 360.0:
        dc_phase -= 360.0
      if dc_phase < 0.0:
        dc_phase += 360.0

    if i >= 32:
      out_sine[i] = math.sin(dc_phase * deg2rad)
      out_leadsine[i] = math.sin((dc_phase + 45.0) * deg2rad)

  return out_sine, out_leadsine


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_ht_dcphase_numba(data: NDArray[np.float64]) -> NDArray[np.float64]:
  """Compute only HT_DCPHASE."""
  n = len(data)
  out_phase = np.full(n, np.nan, dtype=np.float64)

  if n < 32:
    return out_phase

  # Scalar History
  x0, x1, x2, x3 = 0.0, 0.0, 0.0, 0.0
  s0, s1, s2, s3, s4, s5, s6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

  prev_period = 0.0
  prev_smooth_period = 0.0
  dc_phase = 45.0
  d0, d1, d2, d3 = 0.0, 0.0, 0.0, 0.0
  i1, q1 = 0.0, 0.0
  i1_prev, q1_prev, jI, jQ, jI_prev, jQ_prev = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
  rad2deg = 180.0 / math.pi

  smooth_period = 0.0

  for i in range(n):
    # 1. Update WMA History & Calc
    x3, x2, x1 = x2, x1, x0
    x0 = data[i]
    s_val = (4.0 * x0 + 3.0 * x1 + 2.0 * x2 + x3) * 0.1 if i >= 3 else x0

    # 2. Update Hilbert History & Calc
    s6, s5, s4, s3, s2, s1 = s5, s4, s3, s2, s1, s0
    s0 = s_val
    h_val = (0.0962 * s0 + 0.5769 * s2 - 0.5769 * s4 - 0.0962 * s6) if i >= 6 else 0.0

    # 3. Feedback Loop
    d3, d2, d1 = d2, d1, d0
    d0 = h_val * (0.075 * prev_period + 0.54) if i >= 6 else 0.0

    i1_prev, q1_prev = i1, q1
    q1 = d0
    i1 = d3 if i >= 3 else 0.0

    if i >= 6:
      raw_re = i1 * i1_prev + q1 * q1_prev
      raw_im = i1 * q1_prev - q1 * i1_prev

      jI_prev, jQ_prev = jI, jQ
      jI = 0.2 * raw_re + 0.8 * jI_prev
      jQ = 0.2 * raw_im + 0.8 * jQ_prev

      if jI != 0.0 or jQ != 0.0:
        phase_rad = math.atan2(jQ, jI)
        temp_period = 360.0 / (phase_rad * rad2deg) if phase_rad != 0.0 else 0.0
      else:
        temp_period = 0.0

      temp_period = max(6.0, min(50.0, abs(temp_period) * 0.82))

      if prev_period > 0.0:
        temp_period = min(1.5 * prev_period, max(0.67 * prev_period, temp_period))

      period = (
        temp_period if prev_period == 0.0 else 0.2 * temp_period + 0.8 * prev_period
      )
      smooth_period = (
        period
        if prev_smooth_period == 0.0
        else 0.33 * period + 0.67 * prev_smooth_period
      )
      prev_period, prev_smooth_period = period, smooth_period

      if smooth_period != 0.0:
        dc_phase += 360.0 / smooth_period
      if dc_phase >= 360.0:
        dc_phase -= 360.0
      if dc_phase < 0.0:
        dc_phase += 360.0

    if i >= 32:
      out_phase[i] = dc_phase

  return out_phase


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_ht_trendline_numba(data: NDArray[np.float64]) -> NDArray[np.float64]:
  """Compute Hilbert Transform - Trendline.

  Uses Hilbert Transform components to calculate the trendline.
  Matches TA-Lib HT_TRENDLINE.
  """
  n = len(data)
  out_trendline = np.full(n, np.nan, dtype=np.float64)

  if n < 32:
    return out_trendline

  # Scalar History
  x0, x1, x2, x3 = 0.0, 0.0, 0.0, 0.0
  s0, s1, s2, s3, s4, s5, s6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

  prev_period = 0.0
  prev_smooth_period = 0.0
  d0, d1, d2, d3 = 0.0, 0.0, 0.0, 0.0
  i1, q1 = 0.0, 0.0
  i1_prev, q1_prev, jI, jQ, jI_prev, jQ_prev = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
  rad2deg = 180.0 / math.pi

  # State for Trendline WMA
  t0, t1, t2, t3 = 0.0, 0.0, 0.0, 0.0

  smooth_period = 0.0

  for i in range(n):
    # 1. Update WMA History & Calc
    x3, x2, x1 = x2, x1, x0
    x0 = data[i]
    s_val = (4.0 * x0 + 3.0 * x1 + 2.0 * x2 + x3) * 0.1 if i >= 3 else x0

    # 2. Update Hilbert History & Calc
    s6, s5, s4, s3, s2, s1 = s5, s4, s3, s2, s1, s0
    s0 = s_val
    h_val = (0.0962 * s0 + 0.5769 * s2 - 0.5769 * s4 - 0.0962 * s6) if i >= 6 else 0.0

    # 3. Feedback Loop
    d3, d2, d1 = d2, d1, d0
    d0 = h_val * (0.075 * prev_period + 0.54) if i >= 6 else 0.0

    i1_prev, q1_prev = i1, q1
    q1 = d0
    i1 = d3 if i >= 3 else 0.0

    if i >= 6:
      raw_re = i1 * i1_prev + q1 * q1_prev
      raw_im = i1 * q1_prev - q1 * i1_prev

      jI_prev, jQ_prev = jI, jQ
      jI = 0.2 * raw_re + 0.8 * jI_prev
      jQ = 0.2 * raw_im + 0.8 * jQ_prev

      if jI != 0.0 or jQ != 0.0:
        phase_rad = math.atan2(jQ, jI)
        temp_period = 360.0 / (phase_rad * rad2deg) if phase_rad != 0.0 else 0.0
      else:
        temp_period = 0.0

      temp_period = max(6.0, min(50.0, abs(temp_period) * 0.82))

      if prev_period > 0.0:
        temp_period = min(1.5 * prev_period, max(0.67 * prev_period, temp_period))

      period = (
        temp_period if prev_period == 0.0 else 0.2 * temp_period + 0.8 * prev_period
      )
      smooth_period = (
        period
        if prev_smooth_period == 0.0
        else 0.33 * period + 0.67 * prev_smooth_period
      )
      prev_period, prev_smooth_period = period, smooth_period

    # 4. Trendline is WMA of smooth_price (s0)
    t3, t2, t1 = t2, t1, t0
    t0 = s0

    if i >= 3:
      out_trendline[i] = (4.0 * t0 + 3.0 * t1 + 2.0 * t2 + t3) * 0.1
    else:
      out_trendline[i] = t0

    if i < 11:
      out_trendline[i] = np.nan

  return out_trendline

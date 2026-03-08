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
  compute_group_a: bool = True,
  compute_group_b: bool = True,
  compute_trig: bool = True,
  compute_trendline: bool = True,
) -> tuple[
  NDArray[np.float64],
  NDArray[np.float64],
  NDArray[np.float64],
  NDArray[np.float64],
  NDArray[np.float64],
  NDArray[np.float64],
  NDArray[np.float64],
  NDArray[np.float64],
]:
  """Compute Hilbert Transform Cycle components with Component Gating."""
  n = len(data)

  out_period = np.empty(n) if compute_group_a or compute_group_b else np.empty(0)
  out_phase = np.empty(n) if compute_trig else np.empty(0)
  out_inphase = np.empty(n) if compute_group_a or compute_group_b else np.empty(0)
  out_quad = np.empty(n) if compute_group_a or compute_group_b else np.empty(0)
  out_sine = np.empty(n) if compute_trig else np.empty(0)
  out_leadsine = np.empty(n) if compute_trig else np.empty(0)
  out_trendline = np.empty(n) if compute_trendline else np.empty(0)
  out_trendmode = np.empty(n) if compute_trig or compute_trendline else np.empty(0)

  # Initialize outputs with NaN
  if len(out_period) > 0:
    out_period[:] = np.nan
  if len(out_phase) > 0:
    out_phase[:] = np.nan
  if len(out_inphase) > 0:
    out_inphase[:] = np.nan
  if len(out_quad) > 0:
    out_quad[:] = np.nan
  if len(out_sine) > 0:
    out_sine[:] = np.nan
  if len(out_leadsine) > 0:
    out_leadsine[:] = np.nan
  if len(out_trendline) > 0:
    out_trendline[:] = np.nan
  if len(out_trendmode) > 0:
    out_trendmode[:] = np.nan

  if n < 32:
    return (
      out_period,
      out_phase,
      out_inphase,
      out_quad,
      out_sine,
      out_leadsine,
      out_trendline,
      out_trendmode,
    )

  if n < 63 and compute_group_b:
    compute_group_b = False

  # Constants
  a, b = 0.0962, 0.5769
  pi = math.pi
  rad2deg = 180.0 / pi
  deg2rad = pi / 180.0

  # Group A state ... (keep as is) ...
  ga_period = 0.0
  ga_smooth_period = 0.0
  ga_prev_i2, ga_prev_q2 = 0.0, 0.0
  ga_re, ga_im = 0.0, 0.0
  ga_i1e3, ga_i1o3, ga_i1e2, ga_i1o2 = 0.0, 0.0, 0.0, 0.0
  ga_e_buf = np.zeros((4, 3))
  ga_o_buf = np.zeros((4, 3))
  ga_e_p = np.zeros(4)
  ga_o_p = np.zeros(4)
  ga_e_pi = np.zeros(4)
  ga_o_pi = np.zeros(4)

  # Group B state
  gb_period = 0.0
  gb_smooth_period = 0.0
  gb_prev_i2, gb_prev_q2 = 0.0, 0.0
  gb_re, gb_im = 0.0, 0.0
  gb_i1e3, gb_i1o3, gb_i1e2, gb_i1o2 = 0.0, 0.0, 0.0, 0.0
  gb_e_buf = np.zeros((4, 3))
  gb_o_buf = np.zeros((4, 3))
  gb_e_p = np.zeros(4)
  gb_o_p = np.zeros(4)
  gb_e_pi = np.zeros(4)
  gb_o_pi = np.zeros(4)

  # Prefix sums for SMA O(1) - Gated
  prefix_sum = np.zeros(0)
  if compute_trendline:
    prefix_sum = np.empty(n + 1)
    prefix_sum[0] = 0.0
    for i in range(n):
      prefix_sum[i + 1] = prefix_sum[i] + data[i]

  # WMA state
  wma_sum = data[0] + data[1] * 2.0 + data[2] * 3.0
  wma_sub = data[0] + data[1] + data[2]
  trail_idx = 0
  trail_val = 0.0

  smooth_price_buf = np.zeros(50)
  sp_idx = 0

  t1, t2, t3 = 0.0, 0.0, 0.0
  days_in_trend = 0
  dc_phase = 0.0
  sine, lead_sine = 0.0, 0.0

  hil_idx = 0
  for today in range(3, n):
    val = data[today]
    wma_sub += val
    wma_sub -= trail_val
    wma_sum += val * 4.0
    trail_val = data[trail_idx]
    trail_idx += 1
    smoothed = wma_sum * 0.1
    wma_sum -= wma_sub

    # GROUP A Calculation
    if compute_group_a and today >= 12:
      adj_a = 0.075 * ga_period + 0.54
      if today % 2 == 0:
        v = -ga_e_buf[0, hil_idx]
        ga_e_buf[0, hil_idx] = a * smoothed
        v += ga_e_buf[0, hil_idx] - ga_e_p[0] + b * ga_e_pi[0]
        ga_e_p[0] = b * ga_e_pi[0]
        ga_e_pi[0] = smoothed
        det = v * adj_a
        v = -ga_e_buf[1, hil_idx]
        ga_e_buf[1, hil_idx] = a * det
        v += ga_e_buf[1, hil_idx] - ga_e_p[1] + b * ga_e_pi[1]
        ga_e_p[1] = b * ga_e_pi[1]
        ga_e_pi[1] = det
        q1 = v * adj_a
        v = -ga_e_buf[2, hil_idx]
        ga_e_buf[2, hil_idx] = a * ga_i1e3
        v += ga_e_buf[2, hil_idx] - ga_e_p[2] + b * ga_e_pi[2]
        ga_e_p[2] = b * ga_e_pi[2]
        ga_e_pi[2] = ga_i1e3
        ji = v * adj_a
        v = -ga_e_buf[3, hil_idx]
        ga_e_buf[3, hil_idx] = a * q1
        v += ga_e_buf[3, hil_idx] - ga_e_p[3] + b * ga_e_pi[3]
        ga_e_p[3] = b * ga_e_pi[3]
        ga_e_pi[3] = q1
        jq = v * adj_a
        inphase = ga_i1e3
        ga_i1o3, ga_i1o2 = ga_i1o2, det
      else:
        v = -ga_o_buf[0, hil_idx]
        ga_o_buf[0, hil_idx] = a * smoothed
        v += ga_o_buf[0, hil_idx] - ga_o_p[0] + b * ga_o_pi[0]
        ga_o_p[0] = b * ga_o_pi[0]
        ga_o_pi[0] = smoothed
        det = v * adj_a
        v = -ga_o_buf[1, hil_idx]
        ga_o_buf[1, hil_idx] = a * det
        v += ga_o_buf[1, hil_idx] - ga_o_p[1] + b * ga_o_pi[1]
        ga_o_p[1] = b * ga_o_pi[1]
        ga_o_pi[1] = det
        q1 = v * adj_a
        v = -ga_o_buf[2, hil_idx]
        ga_o_buf[2, hil_idx] = a * ga_i1o3
        v += ga_o_buf[2, hil_idx] - ga_o_p[2] + b * ga_o_pi[2]
        ga_o_p[2] = b * ga_o_pi[2]
        ga_o_pi[2] = ga_i1o3
        ji = v * adj_a
        v = -ga_o_buf[3, hil_idx]
        ga_o_buf[3, hil_idx] = a * q1
        v += ga_o_buf[3, hil_idx] - ga_o_p[3] + b * ga_o_pi[3]
        ga_o_p[3] = b * ga_o_pi[3]
        ga_o_pi[3] = q1
        jq = v * adj_a
        inphase = ga_i1o3
        ga_i1e3, ga_i1e2 = ga_i1e2, det

      q2 = 0.2 * (q1 + ji) + 0.8 * ga_prev_q2
      i2 = 0.2 * (inphase - jq) + 0.8 * ga_prev_i2
      ga_re = 0.2 * (i2 * ga_prev_i2 + q2 * ga_prev_q2) + 0.8 * ga_re
      ga_im = 0.2 * (i2 * ga_prev_q2 - q2 * ga_prev_i2) + 0.8 * ga_im
      ga_prev_q2, ga_prev_i2 = q2, i2

      tp = ga_period
      if abs(ga_im) > 1e-12 and abs(ga_re) > 1e-12:
        ga_period = 360.0 / (math.atan(ga_im / ga_re) * rad2deg)
      if ga_period > 1.5 * tp:
        ga_period = 1.5 * tp
      if ga_period < 0.67 * tp:
        ga_period = 0.67 * tp
      if ga_period < 6.0:
        ga_period = 6.0
      elif ga_period > 50.0:
        ga_period = 50.0
      ga_period = 0.2 * ga_period + 0.8 * tp
      ga_smooth_period = 0.33 * ga_period + 0.67 * ga_smooth_period

      if today >= 32:
        if len(out_period) > 0:
          out_period[today] = ga_smooth_period
        if len(out_inphase) > 0:
          out_inphase[today] = inphase
        if len(out_quad) > 0:
          out_quad[today] = q1

    # GROUP B Calculation (Gated)
    if compute_group_b and today >= 37:
      adj_b = 0.075 * gb_period + 0.54
      if today % 2 == 0:
        v = -gb_e_buf[0, hil_idx]
        gb_e_buf[0, hil_idx] = a * smoothed
        v += gb_e_buf[0, hil_idx] - gb_e_p[0] + b * gb_e_pi[0]
        gb_e_p[0] = b * gb_e_pi[0]
        gb_e_pi[0] = smoothed
        det = v * adj_b
        v = -gb_e_buf[1, hil_idx]
        gb_e_buf[1, hil_idx] = a * det
        v += gb_e_buf[1, hil_idx] - gb_e_p[1] + b * gb_e_pi[1]
        gb_e_p[1] = b * gb_e_pi[1]
        gb_e_pi[1] = det
        q1 = v * adj_b
        v = -gb_e_buf[2, hil_idx]
        gb_e_buf[2, hil_idx] = a * gb_i1e3
        v += gb_e_buf[2, hil_idx] - gb_e_p[2] + b * gb_e_pi[2]
        gb_e_p[2] = b * gb_e_pi[2]
        gb_e_pi[2] = gb_i1e3
        ji = v * adj_b
        v = -gb_e_buf[3, hil_idx]
        gb_e_buf[3, hil_idx] = a * q1
        v += gb_e_buf[3, hil_idx] - gb_e_p[3] + b * gb_e_pi[3]
        gb_e_p[3] = b * gb_e_pi[3]
        gb_e_pi[3] = q1
        jq = v * adj_b
        inphase = gb_i1e3
        gb_i1o3, gb_i1o2 = gb_i1o2, det
      else:
        v = -gb_o_buf[0, hil_idx]
        gb_o_buf[0, hil_idx] = a * smoothed
        v += gb_o_buf[0, hil_idx] - gb_o_p[0] + b * gb_o_pi[0]
        gb_o_p[0] = b * gb_o_pi[0]
        gb_o_pi[0] = smoothed
        det = v * adj_b
        v = -gb_o_buf[1, hil_idx]
        gb_o_buf[1, hil_idx] = a * det
        v += gb_o_buf[1, hil_idx] - gb_o_p[1] + b * gb_o_pi[1]
        gb_o_p[1] = b * gb_o_pi[1]
        gb_o_pi[1] = det
        q1 = v * adj_b
        v = -gb_o_buf[2, hil_idx]
        gb_o_buf[2, hil_idx] = a * gb_i1o3
        v += gb_o_buf[2, hil_idx] - gb_o_p[2] + b * gb_o_pi[2]
        gb_o_p[2] = b * gb_o_pi[2]
        gb_o_pi[2] = gb_i1o3
        ji = v * adj_b
        v = -gb_o_buf[3, hil_idx]
        gb_o_buf[3, hil_idx] = a * q1
        v += gb_o_buf[3, hil_idx] - gb_o_p[3] + b * gb_o_pi[3]
        gb_o_p[3] = b * gb_o_pi[3]
        gb_o_pi[3] = q1
        jq = v * adj_b
        inphase = gb_i1o3
        gb_i1e3, gb_i1e2 = gb_i1e2, det

      q2 = 0.2 * (q1 + ji) + 0.8 * gb_prev_q2
      i2 = 0.2 * (inphase - jq) + 0.8 * gb_prev_i2
      gb_re = 0.2 * (i2 * gb_prev_i2 + q2 * gb_prev_q2) + 0.8 * gb_re
      gb_im = 0.2 * (i2 * gb_prev_q2 - q2 * gb_prev_i2) + 0.8 * gb_im
      gb_prev_q2, gb_prev_i2 = q2, i2

      tp = gb_period
      if abs(gb_im) > 1e-12 and abs(gb_re) > 1e-12:
        gb_period = 360.0 / (math.atan(gb_im / gb_re) * rad2deg)
      if gb_period > 1.5 * tp:
        gb_period = 1.5 * tp
      if gb_period < 0.67 * tp:
        gb_period = 0.67 * tp
      if gb_period < 6.0:
        gb_period = 6.0
      elif gb_period > 50.0:
        gb_period = 50.0
      gb_period = 0.2 * gb_period + 0.8 * tp
      gb_smooth_period = 0.33 * gb_period + 0.67 * gb_smooth_period

      dcp = int(gb_smooth_period + 0.5)

      # DCPHASE/SINE/TRENDMODE (Trig Gated)
      if compute_trig:
        smooth_price_buf[sp_idx] = smoothed
        cur_sp_idx = sp_idx
        sp_idx = (sp_idx + 1) % 50

        prev_ph = dc_phase
        rp, ip = 0.0, 0.0
        idx_p = cur_sp_idx
        for i in range(dcp):
          angle = (i * 2.0 * pi) / dcp
          val_p = smooth_price_buf[idx_p]
          rp += math.sin(angle) * val_p
          ip += math.cos(angle) * val_p
          idx_p = (idx_p - 1 + 50) % 50

        # RESTORE Quadrant Logic
        if abs(ip) > 0.0:
          dc_phase = math.atan(rp / ip) * rad2deg
        elif abs(ip) <= 0.01:
          if rp < 0.0:
            dc_phase = -90.0
          elif rp > 0.0:
            dc_phase = 90.0
          else:
            dc_phase = 0.0
        dc_phase += 90.0

        if gb_smooth_period != 0.0:
          dc_phase += 360.0 / gb_smooth_period

        if ip < 0.0:
          dc_phase += 180.0
        if dc_phase > 315.0:
          dc_phase -= 360.0

        prev_s, prev_ls = sine, lead_sine
        sine = math.sin(dc_phase * deg2rad)
        lead_sine = math.sin((dc_phase + 45.0) * deg2rad)

      # TRENDLINE logic
      trendline = 0.0
      if compute_trendline:
        t_sma = data[today]
        if dcp > 0:
          start_idx = today - dcp + 1
          if start_idx < 0:
            start_idx = 0
          count = today - start_idx + 1
          t_sma = (prefix_sum[today + 1] - prefix_sum[start_idx]) / count

        trendline = (4.0 * t_sma + 3.0 * t1 + 2.0 * t2 + t3) * 0.1
        t3, t2, t1 = t2, t1, t_sma

      # TRENDMODE logic (Needs trig)
      trend = 0.0
      if compute_trig:
        trend = 1.0
        if (sine > lead_sine and prev_s <= prev_ls) or (
          sine < lead_sine and prev_s >= prev_ls
        ):
          days_in_trend = 0
          trend = 0.0
        days_in_trend += 1
        if days_in_trend < (0.5 * gb_smooth_period):
          trend = 0.0

        d_phi = dc_phase - prev_ph
        if gb_smooth_period != 0.0:
          if (
            d_phi > 0.67 * 360.0 / gb_smooth_period
            and d_phi < 1.5 * 360.0 / gb_smooth_period
          ):
            trend = 0.0
        if trendline != 0.0 and abs((smoothed - trendline) / trendline) >= 0.015:
          trend = 1.0

      if today >= 63:
        if len(out_phase) > 0:
          out_phase[today] = dc_phase
        if len(out_sine) > 0:
          out_sine[today] = sine
        if len(out_leadsine) > 0:
          out_leadsine[today] = lead_sine
        if len(out_trendline) > 0:
          out_trendline[today] = trendline
        if compute_trig and len(out_trendmode) > 0:
          out_trendmode[today] = trend

    if today % 2 == 0:
      hil_idx = (hil_idx + 1) % 3

  return (
    out_period,
    out_phase,
    out_inphase,
    out_quad,
    out_sine,
    out_leadsine,
    out_trendline,
    out_trendmode,
  )


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_ht_dcperiod_numba(data: NDArray[np.float64]) -> NDArray[np.float64]:
  """Compute only HT_DCPERIOD (optimized with Component Gating)."""
  p, _, _, _, _, _, _, _ = compute_ht_master_numba(
    data,
    compute_group_a=True,
    compute_group_b=False,
    compute_trig=False,
    compute_trendline=False,
  )
  return p


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_ht_phasor_numba(
  data: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  """Compute only HT_PHASOR components."""
  _, _, inphase, quad, _, _, _, _ = compute_ht_master_numba(
    data,
    compute_group_a=True,
    compute_group_b=False,
    compute_trig=False,
    compute_trendline=False,
  )
  return inphase, quad


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_ht_sine_numba(
  data: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  """Compute only HT_SINE components (Sine, LeadSine)."""
  _, _, _, _, sine, lead_sine, _, _ = compute_ht_master_numba(
    data,
    compute_group_a=False,
    compute_group_b=True,
    compute_trig=True,
    compute_trendline=False,
  )
  return sine, lead_sine


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_ht_dcphase_numba(data: NDArray[np.float64]) -> NDArray[np.float64]:
  """Compute only HT_DCPHASE."""
  _, phase, _, _, _, _, _, _ = compute_ht_master_numba(
    data,
    compute_group_a=False,
    compute_group_b=True,
    compute_trig=True,
    compute_trendline=False,
  )
  return phase


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_ht_trendline_numba(data: NDArray[np.float64]) -> NDArray[np.float64]:
  """Compute only HT_TRENDLINE."""
  _, _, _, _, _, _, tl, _ = compute_ht_master_numba(
    data,
    compute_group_a=False,
    compute_group_b=True,
    compute_trig=False,
    compute_trendline=True,
  )
  return tl


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_ht_trendmode_numba(data: NDArray[np.float64]) -> NDArray[np.float64]:
  """Compute only HT_TRENDMODE."""
  _, _, _, _, _, _, _, mode = compute_ht_master_numba(
    data,
    compute_group_a=False,
    compute_group_b=True,
    compute_trig=True,
    compute_trendline=True,
  )
  return mode

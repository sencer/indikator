"""Result types for multi-output indicators.

Uses NamedTuple with .to_pandas() method for zero-copy return.
Includes ResultBase protocol for type checking.
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
import pandas as pd


class ResultBase:
  """Base class for all indicator results."""

  data_index: pd.Index  # pyright: ignore[reportUninitializedInstanceVariable]

  def to_pandas(self) -> pd.Series | pd.DataFrame:
    """Convert result to Pandas object (Series or DataFrame)."""
    raise NotImplementedError


class IndicatorResult(NamedTuple):
  """Generic result for single-output indicators."""

  data_index: pd.Index
  value: NDArray[np.float64]
  name: str

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.value, index=self.data_index, name=self.name, copy=False)


# --- Multi-Output Result Types ---


class ADXResult(NamedTuple):
  data_index: pd.Index
  adx: NDArray[np.float64]
  plus_di: NDArray[np.float64]
  minus_di: NDArray[np.float64]

  def to_pandas(self) -> pd.DataFrame:
    return pd.DataFrame(
      {"adx": self.adx, "plus_di": self.plus_di, "minus_di": self.minus_di},
      index=self.data_index,
      copy=False,
    )


class BollingerBandsResult(NamedTuple):
  data_index: pd.Index
  bb_upper: NDArray[np.float64]
  bb_middle: NDArray[np.float64]
  bb_lower: NDArray[np.float64]

  def to_pandas(self) -> pd.DataFrame:
    return pd.DataFrame(
      {
        "bb_upper": self.bb_upper,
        "bb_middle": self.bb_middle,
        "bb_lower": self.bb_lower,
      },
      index=self.data_index,
      copy=False,
    )


class BollingerResult(NamedTuple):
  data_index: pd.Index
  bb_upper: NDArray[np.float64]
  bb_middle: NDArray[np.float64]
  bb_lower: NDArray[np.float64]
  bb_bandwidth: NDArray[np.float64]
  bb_percent: NDArray[np.float64]

  def to_pandas(self) -> pd.DataFrame:
    return pd.DataFrame(
      {
        "bb_upper": self.bb_upper,
        "bb_middle": self.bb_middle,
        "bb_lower": self.bb_lower,
        "bb_bandwidth": self.bb_bandwidth,
        "bb_percent": self.bb_percent,
      },
      index=self.data_index,
      copy=False,
    )


class MACDResult(NamedTuple):
  data_index: pd.Index
  macd: NDArray[np.float64]
  signal: NDArray[np.float64]
  histogram: NDArray[np.float64]

  def to_pandas(self) -> pd.DataFrame:
    return pd.DataFrame(
      {"macd": self.macd, "signal": self.signal, "histogram": self.histogram},
      index=self.data_index,
      copy=False,
    )


class StochResult(NamedTuple):
  data_index: pd.Index
  stoch_k: NDArray[np.float64]
  stoch_d: NDArray[np.float64]

  def to_pandas(self) -> pd.DataFrame:
    return pd.DataFrame(
      {"stoch_k": self.stoch_k, "stoch_d": self.stoch_d},
      index=self.data_index,
      copy=False,
    )


class AROONResult(NamedTuple):
  data_index: pd.Index
  aroon_up: NDArray[np.float64]
  aroon_down: NDArray[np.float64]
  aroon_osc: NDArray[np.float64]

  def to_pandas(self) -> pd.DataFrame:
    return pd.DataFrame(
      {
        "aroon_up": self.aroon_up,
        "aroon_down": self.aroon_down,
        "aroon_osc": self.aroon_osc,
      },
      index=self.data_index,
      copy=False,
    )


class OpeningRangeResult(NamedTuple):
  data_index: pd.Index
  or_high: NDArray[np.float64]
  or_low: NDArray[np.float64]
  or_mid: NDArray[np.float64]
  or_range: NDArray[np.float64]
  or_breakout: NDArray[np.int8]

  def to_pandas(self) -> pd.DataFrame:
    return pd.DataFrame(
      {
        "or_high": self.or_high,
        "or_low": self.or_low,
        "or_mid": self.or_mid,
        "or_range": self.or_range,
        "or_breakout": self.or_breakout,
      },
      index=self.data_index,
      copy=False,
    )


class PivotPointsResult(NamedTuple):
  data_index: pd.Index
  levels: dict[str, NDArray[np.float64]]

  def to_pandas(self) -> pd.DataFrame:
    return pd.DataFrame(self.levels, index=self.data_index, copy=False)


class ZigzagLegsResult(NamedTuple):
  # Does not have standard index alignment with price data
  # But implements to_pandas() returning the regs DataFrame
  data_index: pd.Index
  direction: NDArray[np.int8]
  start_price: NDArray[np.float64]
  end_price: NDArray[np.float64]
  start_idx: NDArray[np.int64]
  end_idx: NDArray[np.int64]
  pct_change: NDArray[np.float64]

  def to_pandas(self) -> pd.DataFrame:
    return pd.DataFrame(
      {
        "direction": self.direction,
        "start_price": self.start_price,
        "end_price": self.end_price,
        "start_idx": self.start_idx,
        "end_idx": self.end_idx,
        "pct_change": self.pct_change,
      },
      index=self.data_index,
      copy=False,
    )


class StochRSIResult(NamedTuple):
  data_index: pd.Index
  stochrsi_k: NDArray[np.float64]
  stochrsi_d: NDArray[np.float64]

  def to_pandas(self) -> pd.DataFrame:
    return pd.DataFrame(
      {"stochrsi_k": self.stochrsi_k, "stochrsi_d": self.stochrsi_d},
      index=self.data_index,
      copy=False,
    )


class IntradayStatsResult(NamedTuple):
  data_index: pd.Index
  mean: NDArray[np.float64]
  std: NDArray[np.float64]

  def to_pandas(self) -> pd.DataFrame:
    return pd.DataFrame(
      {"mean": self.mean, "std": self.std}, index=self.data_index, copy=False
    )


class MAMAResult(NamedTuple):
  data_index: pd.Index
  mama: NDArray[np.float64]
  fama: NDArray[np.float64]

  def to_pandas(self) -> pd.DataFrame:
    return pd.DataFrame(
      {"mama": self.mama, "fama": self.fama}, index=self.data_index, copy=False
    )


class PhasorResult(NamedTuple):
  data_index: pd.Index
  inphase: NDArray[np.float64]
  quadrature: NDArray[np.float64]

  def to_pandas(self) -> pd.DataFrame:
    return pd.DataFrame(
      {"inphase": self.inphase, "quadrature": self.quadrature},
      index=self.data_index,
      copy=False,
    )


class SineResult(NamedTuple):
  data_index: pd.Index
  sine: NDArray[np.float64]
  leadsine: NDArray[np.float64]

  def to_pandas(self) -> pd.DataFrame:
    return pd.DataFrame(
      {"sine": self.sine, "leadsine": self.leadsine},
      index=self.data_index,
      copy=False,
    )

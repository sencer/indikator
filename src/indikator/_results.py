"""Result types for multi-output indicators.

Uses NamedTuple with .to_pandas() method for zero-copy return.
Includes IndicatorResult protocol for type checking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable

import pandas as pd

if TYPE_CHECKING:
  import numpy as np
  from numpy.typing import NDArray


@runtime_checkable
class IndicatorResult(Protocol):
  """Protocol for all indicator results."""

  index: pd.Index

  def to_pandas(self) -> pd.Series | pd.DataFrame:
    """Convert result to Pandas object (Series or DataFrame)."""
    ...


class EMAResult(NamedTuple):
  index: pd.Index
  ema: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.ema, index=self.index, name="ema", copy=False)


class SMAResult(NamedTuple):
  index: pd.Index
  sma: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.sma, index=self.index, name="sma", copy=False)


class RSIResult(NamedTuple):
  index: pd.Index
  rsi: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.rsi, index=self.index, name="rsi", copy=False)


class SlopeResult(NamedTuple):
  index: pd.Index
  slope: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.slope, index=self.index, name="slope", copy=False)


class ROCResult(NamedTuple):
  index: pd.Index
  roc: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.roc, index=self.index, name="roc", copy=False)


class CMOResult(NamedTuple):
  index: pd.Index
  cmo: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.cmo, index=self.index, name="cmo", copy=False)


class OBVResult(NamedTuple):
  index: pd.Index
  obv: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.obv, index=self.index, name="obv", copy=False)


class TRIXResult(NamedTuple):
  index: pd.Index
  trix: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.trix, index=self.index, name="trix", copy=False)


class WillRResult(NamedTuple):
  index: pd.Index
  willr: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.willr, index=self.index, name="willr", copy=False)


class CCIResult(NamedTuple):
  index: pd.Index
  cci: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.cci, index=self.index, name="cci", copy=False)


class ATRResult(NamedTuple):
  index: pd.Index
  atr: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.atr, index=self.index, name="atr", copy=False)


class ADXSingleResult(NamedTuple):
  index: pd.Index
  adx: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.adx, index=self.index, name="adx", copy=False)


class ADXResult(NamedTuple):
  index: pd.Index
  adx: NDArray[np.float64]
  plus_di: NDArray[np.float64]
  minus_di: NDArray[np.float64]

  def to_pandas(self) -> pd.DataFrame:
    return pd.DataFrame(
      {"adx": self.adx, "plus_di": self.plus_di, "minus_di": self.minus_di},
      index=self.index,
      copy=False,
    )


class BollingerBandsResult(NamedTuple):
  index: pd.Index
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
      index=self.index,
      copy=False,
    )


class BollingerResult(NamedTuple):
  index: pd.Index
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
      index=self.index,
      copy=False,
    )


class MACDResult(NamedTuple):
  index: pd.Index
  macd: NDArray[np.float64]
  signal: NDArray[np.float64]
  histogram: NDArray[np.float64]

  def to_pandas(self) -> pd.DataFrame:
    return pd.DataFrame(
      {"macd": self.macd, "signal": self.signal, "histogram": self.histogram},
      index=self.index,
      copy=False,
    )


class StochResult(NamedTuple):
  index: pd.Index
  stoch_k: NDArray[np.float64]
  stoch_d: NDArray[np.float64]

  def to_pandas(self) -> pd.DataFrame:
    return pd.DataFrame(
      {"stoch_k": self.stoch_k, "stoch_d": self.stoch_d}, index=self.index, copy=False
    )


class AROONResult(NamedTuple):
  index: pd.Index
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
      index=self.index,
      copy=False,
    )


class MFIResult(NamedTuple):
  index: pd.Index
  mfi: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.mfi, index=self.index, name="mfi", copy=False)


class VWAPResult(NamedTuple):
  index: pd.Index
  vwap: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.vwap, index=self.index, name="vwap", copy=False)


class ZScoreResult(NamedTuple):
  index: pd.Index
  zscore: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.zscore, index=self.index, name="zscore", copy=False)


class RVOLResult(NamedTuple):
  index: pd.Index
  rvol: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.rvol, index=self.index, name="rvol", copy=False)


class SectorCorrelationResult(NamedTuple):
  index: pd.Index
  correlation: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.correlation, index=self.index, name="correlation", copy=False)


class ChurnFactorResult(NamedTuple):
  index: pd.Index
  churn: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.churn, index=self.index, name="churn", copy=False)


class OpeningRangeResult(NamedTuple):
  index: pd.Index
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
      index=self.index,
      copy=False,
    )


class PivotPointsResult(NamedTuple):
  index: pd.Index
  levels: dict[str, NDArray[np.float64]]

  def to_pandas(self) -> pd.DataFrame:
    return pd.DataFrame(self.levels, index=self.index, copy=False)


class ZigzagLegsResult(NamedTuple):
  # Does not have standard index alignment with price data
  # But implements to_pandas() returning the regs DataFrame
  index: pd.Index  # RangeIndex usually
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
      index=self.index,
      copy=False,
    )


# --- Tier 1 Indicators ---


class DEMAResult(NamedTuple):
  index: pd.Index
  dema: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.dema, index=self.index, name="dema", copy=False)


class TEMAResult(NamedTuple):
  index: pd.Index
  tema: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.tema, index=self.index, name="tema", copy=False)


class WMAResult(NamedTuple):
  index: pd.Index
  wma: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.wma, index=self.index, name="wma", copy=False)


class KAMAResult(NamedTuple):
  index: pd.Index
  kama: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.kama, index=self.index, name="kama", copy=False)


class SARResult(NamedTuple):
  index: pd.Index
  sar: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.sar, index=self.index, name="sar", copy=False)


class StochRSIResult(NamedTuple):
  index: pd.Index
  stochrsi_k: NDArray[np.float64]
  stochrsi_d: NDArray[np.float64]

  def to_pandas(self) -> pd.DataFrame:
    return pd.DataFrame(
      {"stochrsi_k": self.stochrsi_k, "stochrsi_d": self.stochrsi_d},
      index=self.index,
      copy=False,
    )


class MOMResult(NamedTuple):
  index: pd.Index
  mom: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.mom, index=self.index, name="mom", copy=False)


class NATRResult(NamedTuple):
  index: pd.Index
  natr: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.natr, index=self.index, name="natr", copy=False)


class ULTOSCResult(NamedTuple):
  index: pd.Index
  ultosc: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.ultosc, index=self.index, name="ultosc", copy=False)


class ADResult(NamedTuple):
  index: pd.Index
  ad: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.ad, index=self.index, name="ad", copy=False)


class ADOSCResult(NamedTuple):
  index: pd.Index
  adosc: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.adosc, index=self.index, name="adosc", copy=False)

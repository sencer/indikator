"""Result types for multi-output indicators.

Uses NamedTuple with .to_pandas() method for zero-copy return.
Includes IndicatorResult protocol for type checking.
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
import pandas as pd


class IndicatorResult:
  """Base class for all indicator results."""

  data_index: pd.Index  # pyright: ignore[reportUninitializedInstanceVariable]

  def to_pandas(self) -> pd.Series | pd.DataFrame:
    """Convert result to Pandas object (Series or DataFrame)."""
    raise NotImplementedError


class EMAResult(NamedTuple):
  data_index: pd.Index
  ema: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.ema, index=self.data_index, name="ema", copy=False)


class SMAResult(NamedTuple):
  data_index: pd.Index
  sma: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.sma, index=self.data_index, name="sma", copy=False)


class RSIResult(NamedTuple):
  data_index: pd.Index
  rsi: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.rsi, index=self.data_index, name="rsi", copy=False)


class SlopeResult(NamedTuple):
  data_index: pd.Index
  slope: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.slope, index=self.data_index, name="slope", copy=False)


class ROCResult(NamedTuple):
  data_index: pd.Index
  roc: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.roc, index=self.data_index, name="roc", copy=False)


class CMOResult(NamedTuple):
  data_index: pd.Index
  cmo: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.cmo, index=self.data_index, name="cmo", copy=False)


class OBVResult(NamedTuple):
  data_index: pd.Index
  obv: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.obv, index=self.data_index, name="obv", copy=False)


class TRIXResult(NamedTuple):
  data_index: pd.Index
  trix: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.trix, index=self.data_index, name="trix", copy=False)


class WillRResult(NamedTuple):
  data_index: pd.Index
  willr: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.willr, index=self.data_index, name="willr", copy=False)


class CCIResult(NamedTuple):
  data_index: pd.Index
  cci: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.cci, index=self.data_index, name="cci", copy=False)


class ATRResult(NamedTuple):
  data_index: pd.Index
  atr: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.atr, index=self.data_index, name="atr", copy=False)


class ADXSingleResult(NamedTuple):
  data_index: pd.Index
  adx: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.adx, index=self.data_index, name="adx", copy=False)


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


class MFIResult(NamedTuple):
  data_index: pd.Index
  mfi: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.mfi, index=self.data_index, name="mfi", copy=False)


class VWAPResult(NamedTuple):
  data_index: pd.Index
  vwap: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.vwap, index=self.data_index, name="vwap", copy=False)


class ZScoreResult(NamedTuple):
  data_index: pd.Index
  zscore: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.zscore, index=self.data_index, name="zscore", copy=False)


class RVOLResult(NamedTuple):
  data_index: pd.Index
  rvol: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.rvol, index=self.data_index, name="rvol", copy=False)


class SectorCorrelationResult(NamedTuple):
  data_index: pd.Index
  correlation: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(
      self.correlation, index=self.data_index, name="correlation", copy=False
    )


class ChurnFactorResult(NamedTuple):
  data_index: pd.Index
  churn: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.churn, index=self.data_index, name="churn", copy=False)


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


# --- Tier 1 Indicators ---


class DEMAResult(NamedTuple):
  data_index: pd.Index
  dema: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.dema, index=self.data_index, name="dema", copy=False)


class TEMAResult(NamedTuple):
  data_index: pd.Index
  tema: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.tema, index=self.data_index, name="tema", copy=False)


class WMAResult(NamedTuple):
  data_index: pd.Index
  wma: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.wma, index=self.data_index, name="wma", copy=False)


class KAMAResult(NamedTuple):
  data_index: pd.Index
  kama: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.kama, index=self.data_index, name="kama", copy=False)


class SARResult(NamedTuple):
  data_index: pd.Index
  sar: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.sar, index=self.data_index, name="sar", copy=False)


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


class MOMResult(NamedTuple):
  data_index: pd.Index
  mom: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.mom, index=self.data_index, name="mom", copy=False)


class NATRResult(NamedTuple):
  data_index: pd.Index
  natr: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.natr, index=self.data_index, name="natr", copy=False)


class ULTOSCResult(NamedTuple):
  data_index: pd.Index
  ultosc: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.ultosc, index=self.data_index, name="ultosc", copy=False)


class ADResult(NamedTuple):
  data_index: pd.Index
  ad: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.ad, index=self.data_index, name="ad", copy=False)


class ADOSCResult(NamedTuple):
  data_index: pd.Index
  adosc: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.adosc, index=self.data_index, name="adosc", copy=False)


class TRANGEResult(NamedTuple):
  data_index: pd.Index
  trange: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.trange, index=self.data_index, name="trange", copy=False)


class VWAPAnchoredResult(NamedTuple):
  data_index: pd.Index
  vwap_anchored: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(
      self.vwap_anchored, index=self.data_index, name="vwap_anchored", copy=False
    )


class PlusDMResult(NamedTuple):
  data_index: pd.Index
  plus_dm: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.plus_dm, index=self.data_index, name="plus_dm", copy=False)


class MinusDMResult(NamedTuple):
  data_index: pd.Index
  minus_dm: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.minus_dm, index=self.data_index, name="minus_dm", copy=False)


class PlusDIResult(NamedTuple):
  data_index: pd.Index
  plus_di: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.plus_di, index=self.data_index, name="plus_di", copy=False)


class MinusDIResult(NamedTuple):
  data_index: pd.Index
  minus_di: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.minus_di, index=self.data_index, name="minus_di", copy=False)


class DXResult(NamedTuple):
  data_index: pd.Index
  dx: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.dx, index=self.data_index, name="dx", copy=False)


class ADXRResult(NamedTuple):
  data_index: pd.Index
  adxr: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.adxr, index=self.data_index, name="adxr", copy=False)


class ZScoreIntradayResult(NamedTuple):
  data_index: pd.Index
  zscore_intraday: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(
      self.zscore_intraday, index=self.data_index, name="zscore_intraday", copy=False
    )


class ATRIntradayResult(NamedTuple):
  data_index: pd.Index
  atr_intraday: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(
      self.atr_intraday, index=self.data_index, name="atr_intraday", copy=False
    )


class IntradaySeriesResult(NamedTuple):
  data_index: pd.Index
  values: NDArray[np.float64]
  name: str

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.values, index=self.data_index, name=self.name, copy=False)


class IntradayStatsResult(NamedTuple):
  data_index: pd.Index
  mean: NDArray[np.float64]
  std: NDArray[np.float64]

  def to_pandas(self) -> pd.DataFrame:
    return pd.DataFrame(
      {"mean": self.mean, "std": self.std}, index=self.data_index, copy=False
    )


class BOPResult(NamedTuple):
  data_index: pd.Index
  bop: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.bop, index=self.data_index, name="bop", copy=False)


class TRIMAResult(NamedTuple):
  data_index: pd.Index
  trima: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.trima, index=self.data_index, name="trima", copy=False)


class APOResult(NamedTuple):
  data_index: pd.Index
  apo: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.apo, index=self.data_index, name="apo", copy=False)


class PPOResult(NamedTuple):
  data_index: pd.Index
  ppo: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.ppo, index=self.data_index, name="ppo", copy=False)


class T3Result(NamedTuple):
  data_index: pd.Index
  t3: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.t3, index=self.data_index, name="t3", copy=False)


class ROCPResult(NamedTuple):
  data_index: pd.Index
  rocp: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.rocp, index=self.data_index, name="rocp", copy=False)


class ROCRResult(NamedTuple):
  data_index: pd.Index
  rocr: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.rocr, index=self.data_index, name="rocr", copy=False)


class ROCR100Result(NamedTuple):
  data_index: pd.Index
  rocr100: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.rocr100, index=self.data_index, name="rocr100", copy=False)


class TYPPRICEResult(NamedTuple):
  data_index: pd.Index
  typprice: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.typprice, index=self.data_index, name="typprice", copy=False)


class MEDPRICEResult(NamedTuple):
  data_index: pd.Index
  medprice: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.medprice, index=self.data_index, name="medprice", copy=False)


class WCLPRICEResult(NamedTuple):
  data_index: pd.Index
  wclprice: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.wclprice, index=self.data_index, name="wclprice", copy=False)


class AVGPRICEResult(NamedTuple):
  data_index: pd.Index
  avgprice: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.avgprice, index=self.data_index, name="avgprice", copy=False)


class MIDPRICEResult(NamedTuple):
  data_index: pd.Index
  midprice: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.midprice, index=self.data_index, name="midprice", copy=False)


class MIDPOINTResult(NamedTuple):
  data_index: pd.Index
  midpoint: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.midpoint, index=self.data_index, name="midpoint", copy=False)


class STDDEVResult(NamedTuple):
  data_index: pd.Index
  stddev: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.stddev, index=self.data_index, name="stddev", copy=False)


class VARResult(NamedTuple):
  data_index: pd.Index
  var: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.var, index=self.data_index, name="var", copy=False)


class AROONOSCResult(NamedTuple):
  data_index: pd.Index
  aroonosc: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.aroonosc, index=self.data_index, name="aroonosc", copy=False)


# --- Linear Regression Indicators ---


class LINEARREGResult(NamedTuple):
  data_index: pd.Index
  linearreg: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(
      self.linearreg, index=self.data_index, name="linearreg", copy=False
    )


class LINEARREGInterceptResult(NamedTuple):
  data_index: pd.Index
  linearreg_intercept: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(
      self.linearreg_intercept,
      index=self.data_index,
      name="linearreg_intercept",
      copy=False,
    )


class LINEARREGAngleResult(NamedTuple):
  data_index: pd.Index
  linearreg_angle: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(
      self.linearreg_angle, index=self.data_index, name="linearreg_angle", copy=False
    )


class LINEARREGSlopeResult(NamedTuple):
  data_index: pd.Index
  linearreg_slope: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(
      self.linearreg_slope, index=self.data_index, name="linearreg_slope", copy=False
    )


class TSFResult(NamedTuple):
  data_index: pd.Index
  tsf: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.tsf, index=self.data_index, name="tsf", copy=False)


# --- Statistical Indicators ---


class BETAResult(NamedTuple):
  data_index: pd.Index
  beta: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.beta, index=self.data_index, name="beta", copy=False)


class CORRELResult(NamedTuple):
  data_index: pd.Index
  correl: NDArray[np.float64]

  def to_pandas(self) -> pd.Series:
    return pd.Series(self.correl, index=self.data_index, name="correl", copy=False)


class MAMAResult(NamedTuple):
  data_index: pd.Index
  mama: NDArray[np.float64]
  fama: NDArray[np.float64]

  def to_pandas(self) -> pd.DataFrame:
    return pd.DataFrame(
      {"mama": self.mama, "fama": self.fama}, index=self.data_index, copy=False
    )


# --- Type Aliases ---

type MAResult = (
  SMAResult
  | EMAResult
  | WMAResult
  | DEMAResult
  | TEMAResult
  | TRIMAResult
  | KAMAResult
  | MAMAResult
  | T3Result
)

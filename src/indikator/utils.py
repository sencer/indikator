"""Utility specialized for Indikator results and Pandas integration."""

from __future__ import annotations

from typing import TypeVar, Union

import pandas as pd
import numpy as np

T = TypeVar("T", pd.Series, pd.DataFrame)

def trim_nans(data: T) -> T:
    """Remove leading NaNs from the dataset.
    
    Common for indicators with lookback periods.
    """
    if isinstance(data, pd.DataFrame):
        # Drop rows where any column is NaN
        return data.dropna()
    return data.dropna()

def clean_extreme(data: T, sigma: float = 3.0) -> T:
    """Clamp data within N-sigma bounds."""
    if isinstance(data, pd.DataFrame):
        res = data.copy()
        for col in res.columns:
            m = res[col].mean()
            s = res[col].std()
            res[col] = res[col].clip(m - sigma*s, m + sigma*s)
        return res
    m = data.mean()
    s = data.std()
    return data.clip(m - sigma*s, m + sigma*s)

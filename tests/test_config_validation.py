import numpy as np
import pandas as pd
import pytest

from indikator.atr import atr
from indikator.rsi import rsi
from indikator.vwap import vwap


def test_atr_config_validation():
    """Test that configured ATR function validates input."""
    # Create invalid data (missing required columns)
    invalid_data = pd.DataFrame({"close": [1, 2, 3]})

    made_atr = atr.Config(window=14).make()

    with pytest.raises(ValueError, match="Missing columns"):
        made_atr(invalid_data)


def test_rsi_config_validation():
    """Test that configured RSI function validates input."""
    # Create invalid data (infinite values)
    invalid_data = pd.Series([1.0, np.inf, 3.0])

    made_rsi = rsi.Config(window=14).make()

    with pytest.raises(ValueError, match="must be finite"):
        made_rsi(invalid_data)


def test_vwap_config_validation():
    """Test that configured VWAP function validates input."""
    # Create invalid data (missing volume)
    invalid_data = pd.DataFrame(
        {"high": [10, 11], "low": [9, 10], "close": [9.5, 10.5]},
        index=pd.to_datetime(["2023-01-01", "2023-01-02"]),
    )

    made_vwap = vwap.Config().make()

    with pytest.raises(ValueError, match="Missing columns"):
        made_vwap(invalid_data)

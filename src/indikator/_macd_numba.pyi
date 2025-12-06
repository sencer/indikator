import numpy as np
from numpy.typing import NDArray

def compute_ema_numba(
  prices: NDArray[np.float64],
  window: int,
) -> NDArray[np.float64]: ...
def compute_macd_numba(
  prices: NDArray[np.float64],
  fast_period: int,
  slow_period: int,
  signal_period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...

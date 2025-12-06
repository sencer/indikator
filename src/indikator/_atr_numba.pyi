import numpy as np
from numpy.typing import NDArray

def compute_true_range_numba(
  highs: NDArray[np.float64],
  lows: NDArray[np.float64],
  closes: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def compute_atr_numba(
  true_ranges: NDArray[np.float64],
  window: int,
) -> NDArray[np.float64]: ...

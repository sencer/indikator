import numpy as np
from numpy.typing import NDArray

def compute_rsi_numba(
  prices: NDArray[np.float64],
  window: int,
  epsilon: float = ...,
) -> NDArray[np.float64]: ...

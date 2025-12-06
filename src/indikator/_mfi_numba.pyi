import numpy as np
from numpy.typing import NDArray

def compute_mfi_numba(
  typical_prices: NDArray[np.float64],
  volumes: NDArray[np.float64],
  window: int,
  epsilon: float = ...,
) -> NDArray[np.float64]: ...

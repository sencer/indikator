import numpy as np
from numpy.typing import NDArray

def compute_slope_numba(
  closes: NDArray[np.float64],
  window: int,
) -> NDArray[np.float64]: ...

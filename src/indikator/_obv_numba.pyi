import numpy as np
from numpy.typing import NDArray

def compute_obv_numba(
  closes: NDArray[np.float64],
  volumes: NDArray[np.float64],
) -> NDArray[np.float64]: ...

import numpy as np
from numpy.typing import NDArray

def compute_zigzag_legs_numba(
  closes: NDArray[np.float64],
  threshold: float,
  min_distance_pct: float,
  confirmation_bars: int,
  epsilon: float = ...,
) -> NDArray[np.float64]: ...

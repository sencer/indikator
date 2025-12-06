import numpy as np
from numpy.typing import NDArray

def compute_vwap_numba(
  typical_prices: NDArray[np.float64],
  volumes: NDArray[np.float64],
  reset_mask: NDArray[np.bool_],
) -> NDArray[np.float64]: ...
def compute_anchored_vwap_numba(
  typical_prices: NDArray[np.float64],
  volumes: NDArray[np.float64],
  anchor_index: int,
) -> NDArray[np.float64]: ...

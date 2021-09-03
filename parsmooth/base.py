from typing import NamedTuple
import numpy as np


class MVNParams(NamedTuple):
    mean: np.ndarray
    cov: np.ndarray or None = None
    chol: np.ndarray or None = None


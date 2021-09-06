from functools import partial

import jax
import numpy as np
import pytest

from parsmooth._math_utils import cholesky_update_many


@pytest.mark.parametrize("multiplier", [1., -1.])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_cholesky_update_many(multiplier, seed):
    pass


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_tria(seed):
    pass
import jax
import numpy as np

from parsmooth._base import MVNParams
from parsmooth.sequential._filter import _sqrt_predict, _standard_predict, _sqrt_update, _standard_update
import pytest

from tests._test_utils import get_system


@pytest.fixture(scope="session")
def config():
    jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42, 666])
def test_predict_standard_vs_sqrt(dim_x, seed):
    np.random.seed(seed)
    x, chol_x, F, Q, cholQ, b, _ = get_system(dim_x, dim_x)

    chol_x = _sqrt_predict(F, cholQ, b, chol_x)
    x = _standard_predict(F, Q, b, x)

    np.testing.assert_allclose(x.mean, chol_x.mean, atol=1e-5)
    np.testing.assert_allclose(x.cov, chol_x.chol @ chol_x.chol.T, atol=1e-5)


@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("dim_y", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
def test_update_standard_vs_sqrt(dim_x, dim_y, seed):
    np.random.seed(seed)
    x, chol_x, H, R, cholR, c, y = get_system(dim_x, dim_y)

    x = _standard_update(H, R, c, x, y)
    chol_x = _sqrt_update(H, cholR, c, chol_x, y)

    np.testing.assert_allclose(x.cov, chol_x.chol @ chol_x.chol.T, atol=1e-5)
    np.testing.assert_allclose(x.mean, chol_x.mean, atol=1e-5)

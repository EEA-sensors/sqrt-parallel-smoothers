import jax
import numpy as np

from parsmooth._base import MVNParams
from parsmooth.sequential._filter import _sqrt_predict, _standard_predict, _sqrt_update, _standard_update
import pytest


@pytest.fixture(scope="session")
def config():
    jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42, 666])
def test_predict_standard_vs_sqrt(dim_x, seed):
    np.random.seed(seed)
    # TODO: getting a linear system can be done in a _test_util.py file.
    m = np.random.randn(dim_x)
    cholP = np.random.rand(dim_x, dim_x)
    cholP[np.triu_indices(dim_x, 1)] = 0.
    P = cholP @ cholP.T

    cholQ = np.random.rand(dim_x, dim_x)
    cholQ[np.triu_indices(dim_x, 1)] = 0.
    Q = cholQ @ cholQ.T

    F = np.random.randn(dim_x, dim_x)
    b = np.random.randn(dim_x)

    chol_x = MVNParams(m, None, cholP)
    x = MVNParams(m, P)

    chol_x = _sqrt_predict(F, cholQ, b, chol_x)
    x = _standard_predict(F, Q, b, x)

    np.testing.assert_allclose(x.mean, chol_x.mean, atol=1e-5)
    np.testing.assert_allclose(x.cov, chol_x.chol @ chol_x.chol.T, atol=1e-5)


@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("dim_y", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
def test_update_standard_vs_sqrt(dim_x, dim_y, seed):
    np.random.seed(seed)
    # TODO: getting a linear system can be done in a _test_util.py file.
    m = np.random.randn(dim_x)
    cholP = np.random.rand(dim_x, dim_x)
    cholP[np.triu_indices(dim_x, 1)] = 0.
    P = cholP @ cholP.T

    cholR = np.random.rand(dim_y, dim_y)
    cholR[np.triu_indices(dim_y, 1)] = 0.
    R = cholR @ cholR.T

    H = np.random.randn(dim_y, dim_x)
    c = np.random.randn(dim_y)
    y = np.random.randn(dim_y)

    chol_x = MVNParams(m, None, cholP)
    x = MVNParams(m, P)

    x = _standard_update(H, R, c, x, y)
    chol_x = _sqrt_update(H, cholR, c, chol_x, y)

    np.testing.assert_allclose(x.cov, chol_x.chol @ chol_x.chol.T, atol=1e-5)
    np.testing.assert_allclose(x.mean, chol_x.mean, atol=1e-5)

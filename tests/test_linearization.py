from functools import partial

import jax
import numpy as np
import pytest

from parsmooth._base import MVNParams
from parsmooth.linearization import extended, cubature


@pytest.fixture(scope="session", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)


def linear_function(x, q, a, b, c):
    return a @ x + b @ q + c


@pytest.mark.parametrize("dim_x", [1, 3])
@pytest.mark.parametrize("dim_q", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("method", [extended, cubature])
@pytest.mark.parametrize("sqrt", [True, False])
def test_linear(dim_x, dim_q, seed, method, sqrt):
    if method is extended and sqrt:
        pytest.skip("Not yet implemented")

    np.random.seed(seed)
    a = np.random.randn(dim_x, dim_x)
    b = np.random.randn(dim_x, dim_q)
    c = np.random.randn(dim_x)

    m_x = np.random.randn(dim_x)
    m_q = np.random.randn(dim_q)

    chol_x = np.random.rand(dim_x, dim_x)
    chol_x[np.triu_indices(dim_x, 1)] = 0

    chol_q = np.random.rand(dim_q, dim_q)
    chol_q[np.triu_indices(dim_q, 1)] = 0

    if sqrt:
        x = MVNParams(m_x, None, chol_x)
        q = MVNParams(m_q, None, chol_q)
    else:
        x = MVNParams(m_x, chol_x @ chol_x.T)
        q = MVNParams(m_q, chol_q @ chol_q.T)

    fun = partial(linear_function, a=a, b=b, c=c)

    F_x, F_q, remainder, L = method(fun, x, q, sqrt)

    x_prime = np.random.randn(dim_x)
    q_prime = np.random.randn(dim_q)

    expected = fun(x_prime, q_prime)
    actual = F_x @ (x_prime - m_x) + F_q @ (q_prime - m_q) + remainder

    np.testing.assert_allclose(a, F_x, atol=1e-7)
    np.testing.assert_allclose(b, F_q, atol=1e-7)
    np.testing.assert_allclose(expected, actual, atol=1e-7)

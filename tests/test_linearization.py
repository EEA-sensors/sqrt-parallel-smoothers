from functools import partial

import jax
import numpy as np
import pytest

from parsmooth._base import MVNStandard, FunctionalModel, MVNSqrt
from parsmooth.linearization import extended, cubature


@pytest.fixture(scope="session", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)


def linear_function(x, q, a, b, c):
    return a @ x + b @ q + c


@pytest.mark.parametrize("dim_x", [1, 3])
@pytest.mark.parametrize("dim_q", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("method", [cubature, extended])
@pytest.mark.parametrize("sqrt", [True, False])
def test_linear(dim_x, dim_q, seed, method, sqrt):
    # TODO: use get_system to reduce the number of lines
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
        x = MVNSqrt(m_x, chol_x)
        q = MVNSqrt(m_q, chol_q)
    else:
        x = MVNStandard(m_x, chol_x @ chol_x.T)
        q = MVNStandard(m_q, chol_q @ chol_q.T)

    fun = partial(linear_function, a=a, b=b, c=c)

    fun_model = FunctionalModel(fun, q)
    F_x, Q_lin, remainder = method(fun_model, x)
    if sqrt:
        Q_lin = Q_lin @ Q_lin.T
    x_prime = np.random.randn(dim_x)

    expected = fun(x_prime, m_q)
    actual = F_x @ x_prime + remainder
    expected_Q = (b @ chol_q) @ (b @ chol_q).T

    np.testing.assert_allclose(a, F_x, atol=1e-7)
    np.testing.assert_allclose(expected, actual, atol=1e-7)
    np.testing.assert_allclose(expected_Q, Q_lin, atol=1e-7)


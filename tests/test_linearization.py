from functools import partial

import jax
import numpy as np
import pytest
import jax.numpy as jnp

from parsmooth._base import MVNStandard, FunctionalModel, MVNSqrt, ConditionalMomentsModel
from parsmooth._utils import tria
from parsmooth.linearization import extended, cubature


@pytest.fixture(scope="session", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_debug_nans", False)


def linear_function(x, q, a, b, c):
    return a @ x + b @ q + c


def linear_conditional_mean(x, q, a, b, c):
    return a @ x + b @ q + c


def linear_conditional_cov(_x, b, cov_q):
    return b @ cov_q @ b.T


def linear_conditional_chol(_x, b, chol_q):
    nx, ny = b.shape

    if nx > ny:
        res = jnp.concatenate([b @ chol_q,
                               jnp.zeros((nx, nx - ny))], axis=1)
    else:
        res = tria(b @ chol_q)
    return res


@pytest.mark.parametrize("dim_x", [1, 3])
@pytest.mark.parametrize("dim_q", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("method", [cubature, extended])
@pytest.mark.parametrize("sqrt", [True, False])
def test_linear_functional(dim_x, dim_q, seed, method, sqrt):
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


@pytest.mark.parametrize("dim_x", [1, 3])
@pytest.mark.parametrize("dim_q", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("method", [cubature, extended])
@pytest.mark.parametrize("sqrt", [True, False])
def test_linear_conditional(dim_x, dim_q, seed, method, sqrt):
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

    E_f = partial(linear_conditional_mean, q=m_q, a=a, b=b, c=c)
    V_f = partial(linear_conditional_cov, b=b, cov_q=chol_q @ chol_q.T)
    chol_f = partial(linear_conditional_chol, b=b, chol_q=chol_q)

    if sqrt:
        moments_model = ConditionalMomentsModel(E_f, chol_f)
        x = MVNSqrt(m_x, chol_x)
    else:
        moments_model = ConditionalMomentsModel(E_f, V_f)
        x = MVNStandard(m_x, chol_x @ chol_x.T)

    F_x, Q_lin, remainder = method(moments_model, x)
    if sqrt:
        Q_lin = Q_lin @ Q_lin.T
    x_prime = np.random.randn(dim_x)

    expected = linear_conditional_mean(x_prime, m_q, a, b, c)
    actual = F_x @ x_prime + remainder
    expected_Q = (b @ chol_q) @ (b @ chol_q).T
    np.testing.assert_allclose(a, F_x, atol=1e-3)
    np.testing.assert_allclose(expected, actual, atol=1e-7)
    np.testing.assert_allclose(expected_Q, Q_lin, atol=1e-7)

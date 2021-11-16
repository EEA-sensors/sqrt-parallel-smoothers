from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from parsmooth._base import MVNStandard, FunctionalModel, MVNSqrt, ConditionalMomentsModel
from parsmooth._utils import tria
from parsmooth.linearization import extended, cubature, gauss_hermite, unscented, get_conditional_model

LINEARIZATION_METHODS = [extended, cubature, gauss_hermite, unscented]


@pytest.fixture(scope="session", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_debug_nans", False)


def linear_function(x, a, c):
    return a @ x + c


def linear_conditional_mean(x, q, a, b, c):
    return a @ x + b @ q + c


def linear_conditional_cov(_x, b, cov_q):
    return b @ cov_q @ b.T


def linear_conditional_chol(_x, b, chol_q):
    ny, nq = b.shape
    if ny > nq:
        res = jnp.concatenate([b @ chol_q,
                               jnp.zeros((ny, ny - nq))], axis=1)
    else:
        res = tria(b @ chol_q)
    return res


def transition_mean(x):
    return jnp.log(44.7) + x - jnp.exp(x)


def transition_cov(_x):
    return jnp.array([[0.3 ** 2]])


def transition_chol(_x):
    return jnp.array([[jnp.sqrt(0.3 ** 2)]])


def observation_mean(x, lam):
    return lam * jnp.exp(x)


def observation_cov(x, lam):
    return (lam * jnp.exp(x)).reshape(1, 1)


def observation_chol(x, lam):
    return (jnp.sqrt(lam * jnp.exp(x))).reshape(1, 1)


@pytest.mark.parametrize("dim_x", [1, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("method", LINEARIZATION_METHODS)
@pytest.mark.parametrize("sqrt", [True, False])
def test_linear_functional(dim_x, seed, method, sqrt):
    # TODO: use get_system to reduce the number of lines
    np.random.seed(seed)
    a = np.random.randn(dim_x, dim_x)
    c = np.random.randn(dim_x)

    m_x = np.random.randn(dim_x)
    m_q = np.random.randn(dim_x)

    chol_x = np.random.rand(dim_x, dim_x)
    chol_x[np.triu_indices(dim_x, 1)] = 0

    chol_q = np.random.rand(dim_x, dim_x)
    chol_q[np.triu_indices(dim_x, 1)] = 0

    if sqrt:
        x = MVNSqrt(m_x, chol_x)
        q = MVNSqrt(m_q, chol_q)
    else:
        x = MVNStandard(m_x, chol_x @ chol_x.T)
        q = MVNStandard(m_q, chol_q @ chol_q.T)

    fun = partial(linear_function, a=a, c=c)

    fun_model = FunctionalModel(fun, q)
    F_x, Q_lin, remainder = method(fun_model, x)
    if sqrt:
        Q_lin = Q_lin @ Q_lin.T
    x_prime = np.random.randn(dim_x)

    expected = fun(x_prime) + m_q
    actual = F_x @ x_prime + remainder
    expected_Q = (chol_q) @ (chol_q).T

    np.testing.assert_allclose(a, F_x, atol=1e-7)
    np.testing.assert_allclose(expected, actual, atol=1e-7)
    np.testing.assert_allclose(expected_Q, Q_lin, atol=1e-7)


@pytest.mark.parametrize("dim_x", [1, 3])
@pytest.mark.parametrize("dim_q", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("method", LINEARIZATION_METHODS)
@pytest.mark.parametrize("sqrt", [True, False])
def test_linear_conditional(dim_x, dim_q, seed, method, sqrt):
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


@pytest.mark.parametrize("dim_x", [1, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("method", LINEARIZATION_METHODS)
@pytest.mark.parametrize("test_fun", [jnp.sin, jnp.cos, jnp.exp, jnp.arctan])
def test_sqrt_vs_std(dim_x, seed, method, test_fun):
    np.random.seed(seed)

    m_x = np.random.randn(dim_x)
    m_q = np.random.randn(dim_x)

    chol_x = np.random.rand(dim_x, dim_x)
    chol_x[np.triu_indices(dim_x, 1)] = 0

    chol_q = np.random.rand(dim_x, dim_x)
    chol_q[np.triu_indices(dim_x, 1)] = 0

    chol_x_mvn = MVNSqrt(m_x, chol_x)
    x = MVNStandard(m_x, chol_x @ chol_x.T)

    sqrt_function_model = FunctionalModel(test_fun, chol_x_mvn)
    function_model = FunctionalModel(test_fun, x)

    sqrt_F_x, chol_Q_lin, sqrt_remainder = method(sqrt_function_model, MVNSqrt(m_q, chol_q))
    F_x, Q_lin, remainder = method(function_model, MVNStandard(m_q, chol_q @ chol_q.T))

    np.testing.assert_allclose(sqrt_F_x, F_x, atol=1e-10)
    np.testing.assert_allclose(sqrt_remainder, remainder, atol=1e-10)
    np.testing.assert_allclose(Q_lin, chol_Q_lin @ chol_Q_lin.T, atol=1e-10)


@pytest.mark.parametrize("dim_x", [1])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("method", LINEARIZATION_METHODS)
def test_sqrt_vs_std_conditional_transition(dim_x, seed, method):
    np.random.seed(seed)

    m_x = np.random.randn(dim_x)
    chol_x = np.random.rand(dim_x, dim_x)

    chol_x_mvn = MVNSqrt(m_x, chol_x)
    x_mvn = MVNStandard(m_x, chol_x @ chol_x.T)

    mean_f = partial(transition_mean)
    cov_f = partial(transition_cov)
    chol_f = partial(transition_chol)

    sqrt_moments_model = ConditionalMomentsModel(mean_f, chol_f)
    moments_model = ConditionalMomentsModel(mean_f, cov_f)

    sqrt_F_x, chol_Q_lin, sqrt_remainder = method(sqrt_moments_model, chol_x_mvn)
    F_x, Q_lin, remainder = method(moments_model, x_mvn)

    np.testing.assert_allclose(sqrt_F_x, F_x, atol=1e-10)
    np.testing.assert_allclose(sqrt_remainder, remainder, atol=1e-10)
    np.testing.assert_allclose(Q_lin, chol_Q_lin @ chol_Q_lin.T, atol=1e-10)


@pytest.mark.parametrize("dim_x", [1])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("method", LINEARIZATION_METHODS)
def test_sqrt_vs_std_conditional_observation(dim_x, seed, method):
    np.random.seed(seed)

    m_x = np.random.randn(dim_x)
    chol_x = np.random.rand(dim_x, dim_x)

    chol_x_mvn = MVNSqrt(m_x, chol_x)
    x_mvn = MVNStandard(m_x, chol_x @ chol_x.T)

    mean_h = partial(observation_mean, lam=10)
    cov_h = partial(observation_cov, lam=10)
    chol_h = partial(observation_chol, lam=10)

    sqrt_moments_model = ConditionalMomentsModel(mean_h, chol_h)
    moments_model = ConditionalMomentsModel(mean_h, cov_h)

    sqrt_F_x, chol_Q_lin, sqrt_remainder = method(sqrt_moments_model, chol_x_mvn)
    F_x, Q_lin, remainder = method(moments_model, x_mvn)

    np.testing.assert_allclose(sqrt_F_x, F_x, atol=1e-10)
    np.testing.assert_allclose(sqrt_remainder, remainder, atol=1e-10)
    np.testing.assert_allclose(Q_lin, chol_Q_lin ** 2, atol=1e-10)


@pytest.mark.parametrize("dim_x", [1, 2])
@pytest.mark.parametrize("dim_q", [1, 2])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("method", LINEARIZATION_METHODS)
@pytest.mark.parametrize("sqrt", [True, False])
def test_get_conditional_model(dim_x, dim_q, seed, method, sqrt):
    np.random.seed(seed)
    a = np.random.randn(dim_x, dim_x)
    b = np.random.randn(dim_x, dim_q)
    c = np.random.randn(dim_x)

    f = lambda x, q: a @ x + b @ q + c

    m_x = np.random.randn(dim_x)
    m_q = np.random.randn(dim_q)

    chol_x = np.random.rand(dim_x, dim_x)
    chol_x[np.triu_indices(dim_x, 1)] = 0

    chol_q = np.random.rand(dim_q, dim_q)
    chol_q[np.triu_indices(dim_q, 1)] = 0

    if sqrt:
        x_mvn = MVNSqrt(m_x, chol_x)
        q_mvn = MVNSqrt(m_q, chol_q)
    else:
        x_mvn = MVNStandard(m_x, chol_x @ chol_x.T)
        q_mvn = MVNStandard(m_q, chol_q @ chol_q.T)

    if dim_x != dim_q:
        with pytest.raises(NotImplementedError):
            _ = get_conditional_model(f, q_mvn, method)
        return

    moments_model = get_conditional_model(f, q_mvn, method)

    F_x, Q_lin, remainder = method(moments_model, x_mvn)
    if sqrt:
        Q_lin = Q_lin @ Q_lin.T
    x_prime = np.random.randn(dim_x)

    expected = linear_conditional_mean(x_prime, m_q, a, b, c)
    actual = F_x @ x_prime + remainder
    expected_Q = (b @ chol_q) @ (b @ chol_q).T
    np.testing.assert_allclose(a, F_x, atol=1e-3)
    np.testing.assert_allclose(expected, actual, atol=1e-7)
    np.testing.assert_allclose(expected_Q, Q_lin, atol=1e-7)

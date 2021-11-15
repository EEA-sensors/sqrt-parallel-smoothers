import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax as tfp
from jax.test_util import check_grads

from parsmooth._utils import _cholesky_update, cholesky_update_many, fixed_point, _qr, qr


@pytest.fixture(scope="session", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")


@pytest.mark.parametrize("multiplier", [1., -0.1])
@pytest.mark.parametrize("seed", [0, 42, 666])
@pytest.mark.parametrize("dim_x", [2, 3, 10, 11])
def test_cholesky_update(multiplier, seed, dim_x):
    np.random.seed(seed)
    cholQ = np.random.rand(dim_x, dim_x)
    cholQ[np.triu_indices(dim_x, 1)] = 0.
    v = np.random.randn(dim_x)  # needs to be small enough...

    expected = cholQ @ cholQ.T + multiplier * v[:, None] @ v[None, :]
    eigs_expected = np.linalg.eigvals(expected)
    if min(eigs_expected) <= 1e-6:
        pytest.skip("random vectors do not result in a positive definite matrix.")

    cholRes = _cholesky_update(cholQ, v, multiplier)
    tfpRes = tfp.math.cholesky_update(cholQ, v, multiplier)

    np.testing.assert_allclose(cholRes @ cholRes.T, expected, rtol=1e-4)
    np.testing.assert_allclose(cholRes, tfpRes, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("multiplier", [1., -0.1])
@pytest.mark.parametrize("seed", [0, 1, 2, 42, 666])
@pytest.mark.parametrize("dim_x", [2, 3, 10, 11])
def test_cholesky_update_many(multiplier, seed, dim_x):
    np.random.seed(seed)
    B = 3
    cholQ = np.random.rand(dim_x, dim_x)
    cholQ[np.triu_indices(dim_x, 1)] = 0.
    v = np.random.rand(B, dim_x)  # needs to be small enough...

    expected = cholQ @ cholQ.T + multiplier * sum([v[k, :, None] @ v[k, None, :] for k in range(B)])
    eigs_expected = np.linalg.eigvals(expected)
    if min(eigs_expected) <= 1e-6:
        pytest.skip("random vectors do not result in a positive definite matrix.")

    cholRes = cholesky_update_many(cholQ, v, multiplier)

    np.testing.assert_allclose(cholRes @ cholRes.T, expected, rtol=1e-4)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_qr(seed):
    np.random.seed(seed)
    A = np.random.randn(3, 2)
    B = np.random.randn(2, 3)

    q, r = _qr(jnp.array(A), True)
    np.testing.assert_allclose(q @ r, A)

    q, r = _qr(jnp.array(B), True)
    np.testing.assert_allclose((q @ r), B[:, :2])

    check_grads(qr, (A,), 1, modes=["rev", "fwd"])
    check_grads(qr, (B,), 1, modes=["rev", "fwd"])


def test_fixed_point():
    def my_fun(a, b, x0):
        f = lambda x: (a * x[0] + b[0],)
        return fixed_point(f, x0, lambda i, *_: i < 500)

    actual = my_fun(0.7, (0.5,), (1.,))
    expected = 0.5 / 0.3

    assert actual[0] == pytest.approx(expected, 1e-7, 1e-7)

    check_grads(my_fun, (0.7, (0.5,), (1.,)), 1, modes=["rev"])

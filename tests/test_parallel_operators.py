import jax
import numpy as np
import pytest

from parsmooth.parallel._operators import standard_filtering_operator, sqrt_filtering_operator, \
    standard_smoothing_operator, sqrt_smoothing_operator


@pytest.fixture(scope="session", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_disable_jit', False)
    jax.config.update("jax_debug_nans", False)
    jax.config.update("jax_platform_name", "cpu")


@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
def test_standard_vs_sqrt_filtering_operator(dim_x, seed):
    np.random.seed(seed)
    A1 = np.random.randn(dim_x, dim_x)
    A2 = np.random.randn(dim_x, dim_x)

    b1 = np.random.randn(dim_x)
    b2 = np.random.randn(dim_x)

    U1 = np.random.rand(dim_x, dim_x)
    U1[np.triu_indices(dim_x, 1)] = 0.

    U2 = np.random.rand(dim_x, dim_x)
    U2[np.triu_indices(dim_x, 1)] = 0.

    C1 = U1 @ U1.T
    C2 = U2 @ U2.T

    eta1 = np.random.randn(dim_x)
    eta2 = np.random.randn(dim_x)

    Z1 = np.random.rand(dim_x, dim_x)
    Z1[np.triu_indices(dim_x, 1)] = 0.

    Z2 = np.random.rand(dim_x, dim_x)
    Z2[np.triu_indices(dim_x, 1)] = 0.

    J1 = Z1 @ Z1.T
    J2 = Z2 @ Z2.T

    A_std, b_std, C, eta_std, J = standard_filtering_operator((A1, b1, C1, eta1, J1),
                                                              (A2, b2, C2, eta2, J2))

    A_sqrt, b_sqrt, U, eta_sqrt, Z = sqrt_filtering_operator((A1, b1, U1, eta1, Z1),
                                                             (A2, b2, U2, eta2, Z2))

    np.testing.assert_allclose(A_std, A_sqrt, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(b_std, b_sqrt, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(eta_std, eta_sqrt, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(C, U @ U.T, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(J, Z @ Z.T, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
def test_standard_vs_sqrt_smoothing_operator(dim_x, seed):
    np.random.seed(seed)

    g1 = np.random.randn(dim_x)
    g2 = np.random.randn(dim_x)

    E1 = np.random.randn(dim_x, dim_x)
    E2 = np.random.randn(dim_x, dim_x)

    D1 = np.random.rand(dim_x, dim_x)
    D1[np.triu_indices(dim_x, 1)] = 0.

    D2 = np.random.rand(dim_x, dim_x)
    D2[np.triu_indices(dim_x, 1)] = 0.

    L1 = D1 @ D1.T
    L2 = D2 @ D2.T

    g_std, E_std, L = standard_smoothing_operator((g1, E1, L1),
                                                  (g2, E2, L2))

    g_sqrt, E_sqrt, D = sqrt_smoothing_operator((g1, E1, D1),
                                                (g2, E2, D2))

    np.testing.assert_allclose(g_std, g_sqrt, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(E_std, E_sqrt, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(L, D @ D.T, atol=1e-3, rtol=1e-3)

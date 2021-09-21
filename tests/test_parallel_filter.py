from functools import partial

import jax
import numpy as np
import pytest
from jax.scipy.linalg import solve

from parsmooth._base import FunctionalModel, MVNParams
from parsmooth.linearization import cubature, extended
from parsmooth.linearization._common import fix_mvn
from parsmooth.parallel._filter import _standard_make_associative_filtering_params_first, _sqrt_make_associative_filtering_params_first, \
                                       _standard_make_associative_filtering_params_generic, _sqrt_make_associative_filtering_params_generic
from tests._lgssm import get_data, transition_function as lgssm_f, observation_function as lgssm_h
from tests._test_utils import get_system

LIST_LINEARIZATIONS = [cubature, extended]


@pytest.fixture(scope="session", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_disable_jit', True)
    jax.config.update("jax_debug_nans", False)


@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("dim_y", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("linearization_method", LIST_LINEARIZATIONS)
def test_first_params_standard_vs_sqrt(dim_x, dim_y, seed, linearization_method):
    np.random.seed(seed)

    x0, chol_x0, F, Q, cholQ, b, _ = get_system(dim_x, dim_x)
    x0 = MVNParams(x0.mean, x0.cov, chol_x0.chol)
    _, _, H, R, cholR, c, _ = get_system(dim_x, dim_y)
    m_nominal = np.random.randn(dim_x)
    P_nominal = np.eye(dim_x, dim_x)
    cholP_nominal = P_nominal
    x_nominal = MVNParams(m_nominal, P_nominal, cholP_nominal)
    y = np.random.randn()

    transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNParams(b, Q, cholQ))
    observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNParams(c, R, cholR))

    A_std, b_std, C, eta_std, J = _standard_make_associative_filtering_params_first(linearization_method, transition_model,
                                                                                    observation_model, x_nominal, x0, y, sqrt=False)
    A_sqrt, b_sqrt, U, eta_sqrt, Z = _sqrt_make_associative_filtering_params_first(linearization_method, transition_model,
                                                           observation_model, x_nominal, x0, y, sqrt=True)

    np.testing.assert_allclose(A_std, A_sqrt, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(b_std, b_sqrt, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(eta_std, eta_sqrt, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(C, U @ U.T, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(J, Z @ Z.T, atol=1e-3, rtol=1e-3)

@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("dim_y", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("linearization_method", LIST_LINEARIZATIONS)
def test_generic_params_standard_vs_sqrt(dim_x, dim_y, seed, linearization_method):
    np.random.seed(seed)

    _, _, F, Q, cholQ, b, _ = get_system(dim_x, dim_x)
    _, _, H, R, cholR, c, _ = get_system(dim_x, dim_y)
    m_nominal = np.random.randn(dim_x)
    P_nominal = np.eye(dim_x, dim_x)
    cholP_nominal = P_nominal
    x_nominal = MVNParams(m_nominal, P_nominal, cholP_nominal)
    y = np.random.randn()

    transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNParams(b, Q, cholQ))
    observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNParams(c, R, cholR))

    A_std, b_std, C, eta_std, J = _standard_make_associative_filtering_params_generic(linearization_method, transition_model,
                                                                                    observation_model, x_nominal, y, sqrt=False)
    A_sqrt, b_sqrt, U, eta_sqrt, Z = _sqrt_make_associative_filtering_params_generic(linearization_method, transition_model,
                                                           observation_model, x_nominal, y, sqrt=True)

    np.testing.assert_allclose(A_std, A_sqrt, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(b_std, b_sqrt, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(eta_std, eta_sqrt, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(C, U @ U.T, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(J, Z @ Z.T, atol=1e-3, rtol=1e-3)
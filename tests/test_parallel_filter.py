from functools import partial

import jax
import numpy as np
import pytest

from parsmooth._base import MVNStandard, FunctionalModel, MVNSqrt
from parsmooth.linearization import cubature, extended
from parsmooth.parallel._filter import _standard_make_associative_filtering_params_first, _sqrt_make_associative_filtering_params_first, \
                                       _standard_make_associative_filtering_params_generic, _sqrt_make_associative_filtering_params_generic
from tests._lgssm import transition_function as lgssm_f, observation_function as lgssm_h
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
    x0_sqrt = MVNSqrt(x0.mean, chol_x0.chol)
    x0_std = MVNStandard(x0.mean, x0.cov)
    _, _, H, R, cholR, c, _ = get_system(dim_x, dim_y)
    m_nominal = np.random.randn(dim_x)
    P_nominal = np.eye(dim_x, dim_x)
    cholP_nominal = P_nominal
    x_nominal_sqrt = MVNSqrt(m_nominal, cholP_nominal)
    x_nominal_std = MVNStandard(m_nominal, P_nominal)
    y = np.random.randn()

    sqrt_transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNSqrt(b, cholQ))
    sqrt_observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNSqrt(c, cholR))

    transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNStandard(b, Q))
    observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNStandard(c, R))


    A_std, b_std, C, eta_std, J = _standard_make_associative_filtering_params_first(linearization_method, transition_model,
                                                                                    observation_model, x_nominal_std, x0_std, y)
    A_sqrt, b_sqrt, U, eta_sqrt, Z = _sqrt_make_associative_filtering_params_first(linearization_method, sqrt_transition_model,
                                                           sqrt_observation_model, x_nominal_sqrt, x0_sqrt, y)

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
    x_nominal_sqrt = MVNSqrt(m_nominal, cholP_nominal)
    x_nominal_std = MVNStandard(m_nominal, P_nominal)
    y = np.random.randn()

    sqrt_transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNSqrt(b, cholQ))
    sqrt_observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNSqrt(c, cholR))

    transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNStandard(b, Q))
    observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNStandard(c, R))

    A_std, b_std, C, eta_std, J = _standard_make_associative_filtering_params_generic(linearization_method, transition_model,
                                                                                    observation_model, x_nominal_std, y)
    A_sqrt, b_sqrt, U, eta_sqrt, Z = _sqrt_make_associative_filtering_params_generic(linearization_method, sqrt_transition_model,
                                                           sqrt_observation_model, x_nominal_sqrt, y)

    np.testing.assert_allclose(A_std, A_sqrt, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(b_std, b_sqrt, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(eta_std, eta_sqrt, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(C, U @ U.T, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(J, Z @ Z.T, atol=1e-3, rtol=1e-3)

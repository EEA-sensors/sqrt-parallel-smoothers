from functools import partial

import jax
import numpy as np
import pytest

from parsmooth._base import MVNStandard, FunctionalModel, MVNSqrt
from parsmooth.linearization import cubature, extended
from parsmooth.parallel._filtering import _standard_associative_params_one, _sqrt_associative_params_one, \
    filtering as par_filtering
from parsmooth.sequential._filtering import filtering as seq_filtering
from tests._lgssm import transition_function as lgssm_f, observation_function as lgssm_h, get_data
from tests._test_utils import get_system

LIST_LINEARIZATIONS = [cubature, extended]


@pytest.fixture(scope="session", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_disable_jit', False)
    jax.config.update("jax_debug_nans", False)
    jax.config.update("jax_platform_name", "cpu")


@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("dim_y", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("linearization_method", LIST_LINEARIZATIONS)
def test_params(dim_x, dim_y, seed, linearization_method):
    np.random.seed(seed)

    x0, chol_x0, F, Q, cholQ, b, _ = get_system(dim_x, dim_x)

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

    (A_std, b_std, C, eta_std, J), ssm = _standard_associative_params_one(linearization_method, transition_model,
                                                                          observation_model, x_nominal_std,
                                                                          x_nominal_std,
                                                                          x0.mean, x0.cov, y)
    (A_sqrt, b_sqrt, U, eta_sqrt, Z), sqrt_ssm = _sqrt_associative_params_one(linearization_method,
                                                                              sqrt_transition_model,
                                                                              sqrt_observation_model, x_nominal_sqrt,
                                                                              x_nominal_sqrt, chol_x0.mean,
                                                                              chol_x0.chol, y)

    np.testing.assert_allclose(A_std, A_sqrt, atol=1e-7)
    np.testing.assert_allclose(b_std, b_sqrt, atol=1e-7)
    np.testing.assert_allclose(eta_std, eta_sqrt, atol=1e-7)
    np.testing.assert_allclose(C, U @ U.T, atol=1e-7)
    np.testing.assert_allclose(J, Z @ Z.T, atol=1e-7)

    for actual, expected in zip(ssm, (F, Q, b, H, R, c)):
        np.testing.assert_allclose(actual, expected, atol=1e-7)

    np.testing.assert_allclose(sqrt_ssm[0], F, atol=1e-7)
    np.testing.assert_allclose(sqrt_ssm[1] @ sqrt_ssm[1].T, cholQ @ cholQ.T, atol=1e-7)
    np.testing.assert_allclose(sqrt_ssm[2], b, atol=1e-7)
    np.testing.assert_allclose(sqrt_ssm[3], H, atol=1e-7)
    np.testing.assert_allclose(sqrt_ssm[5], c, atol=1e-7)
    np.testing.assert_allclose(sqrt_ssm[4] @ sqrt_ssm[4].T, cholR @ cholR.T, atol=1e-7)


@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("dim_y", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("linearization_method", LIST_LINEARIZATIONS)
def test_vs_sequential_filter(dim_x, dim_y, seed, linearization_method):
    np.random.seed(seed)
    T = 10

    x0, chol_x0, F, Q, cholQ, b, _ = get_system(dim_x, dim_x)

    _, _, H, R, cholR, c, _ = get_system(dim_x, dim_y)

    m_nominal = np.random.randn(T + 1, dim_x)
    P_nominal = np.repeat(np.eye(dim_x, dim_x)[None, ...], T + 1, axis=0)
    cholP_nominal = P_nominal
    x_nominal_sqrt = MVNSqrt(m_nominal, cholP_nominal)
    x_nominal = MVNStandard(m_nominal, P_nominal)

    true_states, observations = get_data(x0.mean, F, H, R, Q, b, c, T)

    sqrt_transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNSqrt(b, cholQ))
    sqrt_observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNSqrt(c, cholR))

    transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNStandard(b, Q))
    observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNStandard(c, R))

    seq_filter_res, seq_ell = seq_filtering(observations, x0, transition_model, observation_model, linearization_method,
                                            x_nominal, return_loglikelihood=True)
    seq_sqrt_filter_res, seq_sqrt_ell = seq_filtering(observations, chol_x0, sqrt_transition_model,
                                                      sqrt_observation_model,
                                                      linearization_method, x_nominal_sqrt, return_loglikelihood=True)
    par_filter_res, par_ell = par_filtering(observations, x0, transition_model, observation_model, linearization_method,
                                            x_nominal, return_loglikelihood=True)

    sqrt_par_filter_res, par_sqrt_ell = par_filtering(observations, chol_x0, sqrt_transition_model,
                                                      sqrt_observation_model, linearization_method, x_nominal_sqrt,
                                                      return_loglikelihood=True)
    np.testing.assert_array_almost_equal(seq_filter_res.mean, par_filter_res.mean)
    np.testing.assert_array_almost_equal(seq_filter_res.cov, par_filter_res.cov)
    np.testing.assert_array_almost_equal(sqrt_par_filter_res.mean, seq_filter_res.mean)
    np.testing.assert_array_almost_equal(sqrt_par_filter_res.chol @ np.transpose(sqrt_par_filter_res.chol, [0, 2, 1]),
                                         seq_filter_res.cov)
    np.testing.assert_array_almost_equal(seq_filter_res.mean, seq_sqrt_filter_res.mean)

    assert seq_sqrt_ell == pytest.approx(seq_ell)
    assert par_ell == pytest.approx(seq_ell)
    assert par_sqrt_ell == pytest.approx(seq_ell)



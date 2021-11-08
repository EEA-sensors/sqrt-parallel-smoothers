from functools import partial

import jax
import numpy as np
import jax.numpy as jnp
import pytest

from parsmooth._base import MVNStandard, FunctionalModel, MVNSqrt, FunctionalModelX
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

    for actual, expected in zip(sqrt_ssm, (F, cholQ, b, H, cholR, c)):
        np.testing.assert_allclose(actual, expected, atol=1e-7)


@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("linearization_method", LIST_LINEARIZATIONS)
@pytest.mark.parametrize("test_tran_fun", [jnp.sin, jnp.cos, jnp.exp, jnp.arctan])
@pytest.mark.parametrize("test_obs_fun", [jnp.sin, jnp.cos, jnp.exp, jnp.arctan])
def test_params_nonlinear(dim_x, test_tran_fun, test_obs_fun, seed, linearization_method):
    np.random.seed(seed)

    m0 = np.random.randn(dim_x)
    chol0 = np.random.rand(dim_x, dim_x)
    chol0[np.triu_indices(dim_x, 1)] = 0
    x0 = MVNStandard(m0, chol0 @ chol0.T)
    chol_x0 = MVNSqrt(m0, chol0)

    x1 = np.random.randn(dim_x)
    cholx1 = np.random.rand(dim_x, dim_x)
    cholx1[np.triu_indices(dim_x, 1)] = 0
    x_nominal_sqrt1 = MVNSqrt(x1, cholx1)
    x_nominal_std1 = MVNStandard(x1, cholx1 @ cholx1.T)

    x2 = np.random.randn(dim_x)
    cholx2 = np.random.rand(dim_x, dim_x)
    cholx2[np.triu_indices(dim_x, 1)] = 0
    x_nominal_sqrt2 = MVNSqrt(x2, cholx2)
    x_nominal_std2 = MVNStandard(x2, cholx2 @ cholx2.T)

    x3 = np.random.randn(dim_x)
    cholx3 = np.random.rand(dim_x, dim_x)
    cholx3[np.triu_indices(dim_x, 1)] = 0
    x_tran_sqrt = MVNSqrt(x3, cholx3)
    x_tran_std = MVNStandard(x3, cholx3 @ cholx3.T)

    x4 = np.random.randn(dim_x)
    cholx4 = np.random.rand(dim_x, dim_x)
    cholx4[np.triu_indices(dim_x, 1)] = 0
    x_obs_sqrt = MVNSqrt(x4, cholx4)
    x_obs_std = MVNStandard(x4, cholx4 @ cholx4.T)
    y = np.random.randn()

    sqrt_transition_model = FunctionalModelX(test_tran_fun, x_tran_sqrt)
    transition_model = FunctionalModelX(test_tran_fun, x_tran_std)

    sqrt_observation_model = FunctionalModelX(test_obs_fun, x_obs_sqrt)
    observation_model = FunctionalModelX(test_obs_fun, x_obs_std)

    (A_std, b_std, C, eta_std, J), ssm = _standard_associative_params_one(linearization_method,
                                                                          transition_model,
                                                                          observation_model,
                                                                          x_nominal_std1,
                                                                          x_nominal_std2,
                                                                          x0.mean,
                                                                          x0.cov,
                                                                          y)

    (A_sqrt, b_sqrt, U, eta_sqrt, Z), sqrt_ssm = _sqrt_associative_params_one(linearization_method,
                                                                              sqrt_transition_model,
                                                                              sqrt_observation_model,
                                                                              x_nominal_sqrt1,
                                                                              x_nominal_sqrt2,
                                                                              chol_x0.mean,
                                                                              chol_x0.chol,
                                                                              y)

    np.testing.assert_allclose(A_std, A_sqrt, atol=1e-7)
    np.testing.assert_allclose(b_std, b_sqrt, atol=1e-7)
    np.testing.assert_allclose(eta_std, eta_sqrt, atol=1e-7)
    np.testing.assert_allclose(C, U @ U.T, atol=1e-7)
    np.testing.assert_allclose(J, Z @ Z.T, atol=1e-7)


@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("dim_y", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("linearization_method", LIST_LINEARIZATIONS)
def test_vs_sequential_filter(dim_x, dim_y, seed, linearization_method):
    np.random.seed(seed)
    T = 15

    x0, chol_x0, F, Q, cholQ, b, _ = get_system(dim_x, dim_x)

    _, _, H, R, cholR, c, _ = get_system(dim_x, dim_y)

    m_nominal = np.random.randn(T, dim_x)
    P_nominal = np.repeat(np.eye(dim_x, dim_x)[None, ...], T, axis=0)
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
    np.testing.assert_array_almost_equal(seq_sqrt_filter_res.mean, sqrt_par_filter_res.mean)
    np.testing.assert_array_almost_equal(seq_filter_res.mean, seq_sqrt_filter_res.mean)
    np.testing.assert_array_almost_equal(seq_filter_res.cov, par_filter_res.cov)
    np.testing.assert_array_almost_equal(sqrt_par_filter_res.mean, seq_filter_res.mean)
    np.testing.assert_array_almost_equal(sqrt_par_filter_res.chol @ np.transpose(sqrt_par_filter_res.chol, [0, 2, 1]),
                                         seq_filter_res.cov)

    assert par_sqrt_ell == pytest.approx(par_ell)
    assert seq_sqrt_ell == pytest.approx(seq_ell)
    assert par_sqrt_ell == pytest.approx(seq_sqrt_ell)
    assert par_ell == pytest.approx(seq_ell)



@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("seed", [2])
@pytest.mark.parametrize("test_tran_fun", [jnp.sin, jnp.cos, jnp.exp, jnp.arctan])
@pytest.mark.parametrize("test_obs_fun", [jnp.sin])
@pytest.mark.parametrize("linearization_method", LIST_LINEARIZATIONS)
def test_vs_sequential_filter_nonlinear(dim_x, test_tran_fun, test_obs_fun, seed, linearization_method):
    np.random.seed(seed)
    T = 15

    m0 = np.random.randn(dim_x)
    chol0 = np.random.rand(dim_x, dim_x)
    chol0[np.triu_indices(dim_x, 1)] = 0
    x0 = MVNStandard(m0, chol0 @ chol0.T)
    chol_x0 = MVNSqrt(m0, chol0)

    m_nominal = np.random.randn(T, dim_x)
    P_nominal = np.repeat(np.eye(dim_x, dim_x)[None, ...], T, axis=0)
    cholP_nominal = P_nominal
    x_nominal_sqrt = MVNSqrt(m_nominal, cholP_nominal)
    x_nominal = MVNStandard(m_nominal, P_nominal)

    x3 = np.random.randn(dim_x)
    cholx3 = np.random.rand(dim_x, dim_x)
    cholx3[np.triu_indices(dim_x, 1)] = 0
    x_tran_sqrt = MVNSqrt(x3, cholx3)
    x_tran_std = MVNStandard(x3, cholx3 @ cholx3.T)

    x4 = np.random.randn(dim_x)
    cholx4 = np.random.rand(dim_x, dim_x)
    cholx4[np.triu_indices(dim_x, 1)] = 0
    x_obs_sqrt = MVNSqrt(x4, cholx4)
    x_obs_std = MVNStandard(x4, cholx4 @ cholx4.T)
    observations = np.random.randn(T, dim_x)

    sqrt_transition_model = FunctionalModelX(test_tran_fun, x_tran_sqrt)
    transition_model = FunctionalModelX(test_tran_fun, x_tran_std)
    sqrt_observation_model = FunctionalModelX(test_obs_fun, x_obs_sqrt)
    observation_model = FunctionalModelX(test_obs_fun, x_obs_std)

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
    np.testing.assert_array_almost_equal(seq_sqrt_filter_res.mean, sqrt_par_filter_res.mean)
    np.testing.assert_array_almost_equal(seq_filter_res.mean, seq_sqrt_filter_res.mean)
    np.testing.assert_array_almost_equal(seq_filter_res.cov, par_filter_res.cov)
    np.testing.assert_array_almost_equal(sqrt_par_filter_res.mean, seq_filter_res.mean)
    np.testing.assert_array_almost_equal(sqrt_par_filter_res.chol @ np.transpose(sqrt_par_filter_res.chol, [0, 2, 1]),
                                         seq_filter_res.cov)

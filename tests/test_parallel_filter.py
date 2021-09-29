from functools import partial

import jax
import numpy as np
import pytest

from parsmooth._base import MVNStandard, FunctionalModel, MVNSqrt
from parsmooth.linearization import cubature, extended
from parsmooth.parallel._filter import _standard_associative_params_one, _sqrt_associative_params_one, \
    filtering as par_filtering, _standard_associative_params, _sqrt_associative_params
from parsmooth.sequential._filter import filtering as seq_filtering
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
def test_params_standard_vs_sqrt(dim_x, dim_y, seed, linearization_method):
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

    A_std, b_std, C, eta_std, J = _standard_associative_params_one(linearization_method, transition_model,
                                                                   observation_model, x_nominal_std, x_nominal_std,
                                                                   x0.mean, x0.cov, y)
    A_sqrt, b_sqrt, U, eta_sqrt, Z = _sqrt_associative_params_one(linearization_method, sqrt_transition_model,
                                                                  sqrt_observation_model, x_nominal_sqrt,
                                                                  x_nominal_sqrt, chol_x0.mean,
                                                                  chol_x0.chol, y)

    np.testing.assert_allclose(A_std, A_sqrt)
    np.testing.assert_allclose(b_std, b_sqrt)
    np.testing.assert_allclose(eta_std, eta_sqrt)
    np.testing.assert_allclose(C, U @ U.T)
    np.testing.assert_allclose(J, Z @ Z.T)


@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("dim_y", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("linearization_method", LIST_LINEARIZATIONS)
def test_all_associative_params(dim_x, dim_y, seed, linearization_method):
    np.random.seed(seed)
    T = 4

    x0, chol_x0, F, Q, cholQ, b, _ = get_system(dim_x, dim_x)

    _, _, H, R, cholR, c, _ = get_system(dim_x, dim_y)

    m_nominal = np.random.randn(T + 1, dim_x)
    P_nominal = np.repeat(np.eye(dim_x, dim_x)[None, ...], T + 1, axis=0)
    cholP_nominal = P_nominal
    x_nominal_sqrt = MVNSqrt(m_nominal, cholP_nominal)
    x_nominal = MVNStandard(m_nominal, P_nominal)

    _, observations = get_data(x0.mean, F, H, R, Q, b, c, T)

    sqrt_transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNSqrt(b, cholQ))
    sqrt_observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNSqrt(c, cholR))

    transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNStandard(b, Q))
    observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNStandard(c, R))

    standard_params = _standard_associative_params(linearization_method, transition_model, observation_model, x_nominal,
                                                   x0, observations)
    sqrt_params = _sqrt_associative_params(linearization_method, sqrt_transition_model, sqrt_observation_model,
                                           x_nominal_sqrt, chol_x0, observations)

    np.testing.assert_array_almost_equal(standard_params[0], sqrt_params[0])
    np.testing.assert_array_almost_equal(standard_params[1], sqrt_params[1])
    np.testing.assert_array_almost_equal(standard_params[2], sqrt_params[2] @ np.transpose(sqrt_params[2], [0, 2, 1]))
    np.testing.assert_array_almost_equal(standard_params[3], sqrt_params[3])
    np.testing.assert_array_almost_equal(standard_params[4], sqrt_params[4] @ np.transpose(sqrt_params[4], [0, 2, 1]))


@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("dim_y", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("linearization_method", LIST_LINEARIZATIONS)
def test_vs_sequential_filter(dim_x, dim_y, seed, linearization_method):
    np.random.seed(seed)
    T = 15

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

    seq_filter_res = seq_filtering(observations, x0, transition_model, observation_model, linearization_method,
                                   x_nominal)
    par_filter_res = par_filtering(observations, x0, transition_model, observation_model, linearization_method,
                                   x_nominal)

    sqrt_par_filter_res = par_filtering(observations, chol_x0, sqrt_transition_model, sqrt_observation_model,
                                        linearization_method, x_nominal_sqrt)

    np.testing.assert_array_almost_equal(seq_filter_res.mean, par_filter_res.mean)
    np.testing.assert_array_almost_equal(seq_filter_res.cov, par_filter_res.cov)
    np.testing.assert_array_almost_equal(sqrt_par_filter_res.mean, seq_filter_res.mean)
    np.testing.assert_array_almost_equal(sqrt_par_filter_res.chol @ np.transpose(sqrt_par_filter_res.chol, [0, 2, 1]),
                                         seq_filter_res.cov)

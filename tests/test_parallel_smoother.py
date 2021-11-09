from functools import partial

import jax
import numpy as np
import pytest

from parsmooth._base import MVNStandard, FunctionalModel, MVNSqrt
from parsmooth.linearization import cubature, extended
from parsmooth.parallel._smoothing import _standard_associative_params, _sqrt_associative_params, \
    smoothing as par_smoothing
from parsmooth.sequential._filtering import filtering as seq_filtering
from parsmooth.sequential._smoothing import smoothing as seq_smoothing
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

    transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNStandard(b, Q))

    g_std, E_std, L = _standard_associative_params(linearization_method, transition_model, x_nominal_std, x0.mean,
                                                   x0.cov)
    g_sqrt, E_sqrt, D = _sqrt_associative_params(linearization_method, sqrt_transition_model, x_nominal_sqrt,
                                                 chol_x0.mean, chol_x0.chol)

    np.testing.assert_allclose(g_std, g_sqrt)
    np.testing.assert_allclose(E_std, E_sqrt)
    np.testing.assert_allclose(L, D @ D.T)


@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("dim_y", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("linearization_method", LIST_LINEARIZATIONS)
def test_vs_sequential_smoother(dim_x, dim_y, seed, linearization_method):
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

    seq_filter_res = seq_filtering(observations, x0, transition_model, observation_model, linearization_method,
                                   x_nominal)
    seq_smoother_res = seq_smoothing(transition_model, seq_filter_res, linearization_method)
    par_smoother_res = par_smoothing(transition_model, seq_filter_res, linearization_method)

    seq_sqrt_filter_res = seq_filtering(observations, chol_x0, sqrt_transition_model, sqrt_observation_model,
                                        linearization_method,
                                        x_nominal_sqrt)
    seq_sqrt_smoother_res = seq_smoothing(sqrt_transition_model, seq_sqrt_filter_res, linearization_method)
    par_sqrt_smoother_res = par_smoothing(sqrt_transition_model, seq_sqrt_filter_res, linearization_method)

    np.testing.assert_array_almost_equal(seq_smoother_res.mean, par_smoother_res.mean, decimal=4)
    np.testing.assert_array_almost_equal(seq_smoother_res.cov, par_smoother_res.cov, decimal=4)
    np.testing.assert_array_almost_equal(seq_sqrt_smoother_res.mean, par_sqrt_smoother_res.mean, decimal=4)
    np.testing.assert_array_almost_equal(
        seq_sqrt_smoother_res.chol @ np.transpose(seq_sqrt_smoother_res.chol, [0, 2, 1]),
        seq_smoother_res.cov, decimal=4)

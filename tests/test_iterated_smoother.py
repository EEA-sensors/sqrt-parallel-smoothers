from functools import partial

import jax
import numpy as np
import pytest
from jax.test_util import check_grads

from parsmooth._base import MVNStandard, FunctionalModel, MVNSqrt
from parsmooth.linearization import cubature, extended
from parsmooth.methods import iterated_smoothing
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


@pytest.mark.parametrize("dim_x", [1, 2])
@pytest.mark.parametrize("dim_y", [1, 2])
@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("linearization_method", LIST_LINEARIZATIONS)
@pytest.mark.parametrize("parallel", [True, False])
def test_linear(dim_x, dim_y, seed, linearization_method, parallel):
    np.random.seed(seed)
    T = 5

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

    iterated_res = iterated_smoothing(observations, x0, transition_model, observation_model, linearization_method,
                                      x_nominal, parallel, criterion=lambda i, *_: i < 5)

    sqrt_iterated_res = iterated_smoothing(observations, chol_x0, sqrt_transition_model, sqrt_observation_model,
                                           linearization_method, x_nominal_sqrt, parallel,
                                           criterion=lambda i, *_: i < 5)

    np.testing.assert_array_almost_equal(iterated_res.mean, seq_smoother_res.mean, decimal=4)
    np.testing.assert_array_almost_equal(iterated_res.cov, seq_smoother_res.cov, decimal=4)
    np.testing.assert_array_almost_equal(sqrt_iterated_res.mean, seq_smoother_res.mean, decimal=4)
    np.testing.assert_array_almost_equal(
        sqrt_iterated_res.chol @ np.transpose(sqrt_iterated_res.chol, [0, 2, 1]),
        seq_smoother_res.cov, decimal=4)


@pytest.mark.skip("Not sure if gradient rule works for linear function in sigma points case.")
@pytest.mark.parametrize("dim_x", [1])
@pytest.mark.parametrize("dim_y", [1])
@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("linearization_method", LIST_LINEARIZATIONS)
@pytest.mark.parametrize("parallel", [True, False])
def test_linear_gradient(dim_x, dim_y, seed, linearization_method, parallel):
    np.random.seed(seed)
    T = 5

    x0, chol_x0, F, Q, cholQ, b, _ = get_system(dim_x, dim_x)

    _, _, H, R, cholR, c, _ = get_system(dim_x, dim_y)

    m_nominal = np.random.randn(T + 1, dim_x)
    P_nominal = np.repeat(np.eye(dim_x, dim_x)[None, ...], T + 1, axis=0)
    cholP_nominal = P_nominal
    x_nominal_sqrt = MVNSqrt(m_nominal, cholP_nominal)
    x_nominal = MVNStandard(m_nominal, P_nominal)

    true_states, observations = get_data(x0.mean, F, H, R, Q, b, c, T)

    @jax.jit
    def my_fun(A, B, x0, nominal):
        if isinstance(x0, MVNSqrt):
            transition_model = FunctionalModel(partial(lgssm_f, A=A), MVNSqrt(b, cholQ))
            observation_model = FunctionalModel(partial(lgssm_h, H=B), MVNSqrt(c, cholR))
        else:
            transition_model = FunctionalModel(partial(lgssm_f, A=A), MVNStandard(b, Q))
            observation_model = FunctionalModel(partial(lgssm_h, H=B), MVNStandard(c, R))

        iterated_res = iterated_smoothing(observations, x0, transition_model, observation_model,
                                          linearization_method, nominal, parallel, criterion=lambda i, *_: i < 5)
        return iterated_res

    check_grads(my_fun, (F, H, chol_x0, x_nominal_sqrt), 1, ["rev"], atol=1e-3, rtol=1e-3)
    check_grads(my_fun, (F, H, x0, x_nominal), 1, ["rev"], atol=1e-3, rtol=1e-3)

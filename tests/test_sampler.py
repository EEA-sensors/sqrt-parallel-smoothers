from functools import partial

import jax
import numpy as np
import pytest

from parsmooth._base import FunctionalModel, MVNStandard, MVNSqrt
from parsmooth.linearization import cubature, extended
from parsmooth.sequential._filter import filtering
from parsmooth.sequential._pathwise_sampler import sampler
from parsmooth.sequential._smoother import smoother
from tests._lgssm import get_data, transition_function as lgssm_f, observation_function as lgssm_h
from tests._test_utils import get_system

LIST_LINEARIZATIONS = [cubature, extended]


@pytest.fixture(scope="session", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_disable_jit', False)
    jax.config.update("jax_debug_nans", False)


@pytest.mark.parametrize("dim_x", [1, 3])
@pytest.mark.parametrize("dim_y", [2, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("linearization", LIST_LINEARIZATIONS)
@pytest.mark.parametrize("jax_seed", [123, 666])
def test_samples_marginals(dim_x, dim_y, seed, linearization, jax_seed):
    np.random.seed(seed)
    key = jax.random.PRNGKey(jax_seed)

    T = 10
    N = 1_000

    x0, chol_x0, F, Q, cholQ, b, _ = get_system(dim_x, dim_x)
    _, _, H, R, cholR, c, _ = get_system(dim_x, dim_y)

    _, ys = get_data(x0.mean, F, H, R, Q, b, c, T)
    sqrt_transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNSqrt(b, cholQ))
    sqrt_observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNSqrt(c, cholR))

    transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNStandard(b, Q))
    observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNStandard(c, R))

    for method in LIST_LINEARIZATIONS:
        filtered_states = filtering(ys, x0, transition_model, observation_model, method)
        smoothed_states = smoother(transition_model, filtered_states, method)
        samples = sampler(key, N, transition_model, filtered_states, method, smoothed_states)

        sqrt_filtered_states = filtering(ys, chol_x0, sqrt_transition_model, sqrt_observation_model, method)
        sqrt_smoothed_states = smoother(sqrt_transition_model, sqrt_filtered_states, method)
        sqrt_samples = sampler(key, N, sqrt_transition_model, sqrt_filtered_states, method, sqrt_smoothed_states)

        np.testing.assert_allclose(samples, sqrt_samples)
        np.testing.assert_allclose(samples.mean(-1), smoothed_states.mean, rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(sqrt_samples.mean(-1), smoothed_states.mean, rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(samples.var(-1), np.diagonal(smoothed_states.cov, axis1=1, axis2=2),
                                   rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(sqrt_samples.std(-1), np.diagonal(np.abs(sqrt_smoothed_states.chol),
                                                                     axis1=1, axis2=2), rtol=1e-2, atol=1e-2)

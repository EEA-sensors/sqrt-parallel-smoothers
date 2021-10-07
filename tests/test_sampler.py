from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import kstest

from parsmooth._base import FunctionalModel, MVNStandard, MVNSqrt
from parsmooth.linearization import cubature, extended
from parsmooth.methods import sampling
from parsmooth.sequential._filtering import filtering
from parsmooth.sequential._smoothing import smoothing
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
@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("linearization", LIST_LINEARIZATIONS)
@pytest.mark.parametrize("jax_seed", [123])
@pytest.mark.parametrize("parallel", [False, True])
def test_samples_marginals(dim_x, dim_y, seed, linearization, jax_seed, parallel):
    np.random.seed(seed)
    key = jax.random.PRNGKey(jax_seed)

    T = 10
    N = 100_000

    x0, chol_x0, F, Q, cholQ, b, _ = get_system(dim_x, dim_x)
    _, _, H, R, cholR, c, _ = get_system(dim_x, dim_y)

    _, ys = get_data(x0.mean, F, H, R, Q, b, c, T)
    sqrt_transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNSqrt(b, cholQ))

    transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNStandard(b, Q))
    observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNStandard(c, R))

    for method in LIST_LINEARIZATIONS:
        filtered_states = filtering(ys, x0, transition_model, observation_model, method)
        smoothed_states = smoothing(transition_model, filtered_states, method)
        samples = sampling(key, N, transition_model, filtered_states, method, smoothed_states, parallel=parallel)

        sqrt_filtered_states = MVNSqrt(filtered_states.mean, jax.vmap(jnp.linalg.cholesky)(filtered_states.cov))
        sqrt_smoothed_states = MVNSqrt(smoothed_states.mean, jax.vmap(jnp.linalg.cholesky)(smoothed_states.cov))
        sqrt_samples = sampling(key, N, sqrt_transition_model, sqrt_filtered_states, method, sqrt_smoothed_states,
                                parallel=parallel)

        print(kstest(samples[0][:, 0], "norm", (smoothed_states.mean[0, 0], smoothed_states.cov[0, 0, 0])))

        np.testing.assert_allclose(samples.mean(1), smoothed_states.mean, rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(samples.var(1), np.diagonal(smoothed_states.cov, axis1=1, axis2=2),
                                   rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(sqrt_samples.mean(1), sqrt_smoothed_states.mean,
                                   rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(sqrt_samples.var(1), np.diagonal(smoothed_states.cov, axis1=1, axis2=2),
                                   rtol=1e-2, atol=1e-2)

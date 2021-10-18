import jax
import jax.numpy as jnp
import numpy as np
import pytest

from parsmooth._base import MVNStandard, FunctionalModel, MVNSqrt
from parsmooth.linearization import cubature, extended
from parsmooth.methods import iterated_smoothing
from tests.bearings.bearings_utils import make_parameters

LIST_LINEARIZATIONS = [cubature]


@pytest.fixture(scope="session", autouse=True)
def config():
    jax.config.update("jax_enable_x64", False)
    jax.config.update('jax_disable_jit', False)
    jax.config.update("jax_debug_nans", False)


@pytest.mark.skip("Skip on continuous integration")
@pytest.mark.parametrize("linearization_method", LIST_LINEARIZATIONS)
@pytest.mark.parametrize("parallel", [True, False])
def test_bearings(linearization_method, parallel):
    s1 = jnp.array([-1.5, 0.5])  # First sensor location
    s2 = jnp.array([1., 1.])  # Second sensor location
    r = 0.5  # Observation noise (stddev)
    dt = 0.01  # discretization time step
    qc = 0.01  # discretization noise
    qw = 0.1  # discretization noise

    ys = np.load("./bearings/ys.npy")
    if linearization_method is extended:
        with np.load("./bearings//ieks.npz") as loaded:
            expected_mean, expected_cov = loaded["arr_0"], loaded["arr_1"]
    elif linearization_method is cubature:
        with np.load("./bearings//icks.npz") as loaded:
            expected_mean, expected_cov = loaded["arr_0"], loaded["arr_1"]
    else:
        pytest.skip("We don't have regression data for this linearization")

    Q, R, observation_function, transition_function = make_parameters(qc, qw, r, dt, s1, s2)

    m0 = jnp.array([-1., -1., 0., 0., 0.])
    chol_P0 = P0 = jnp.eye(5)

    chol_Q = jnp.linalg.cholesky(Q)
    chol_R = jnp.linalg.cholesky(R)

    init = MVNStandard(m0, P0)
    chol_init = MVNSqrt(m0, chol_P0)

    sqrt_transition_model = FunctionalModel(transition_function, MVNSqrt(jnp.zeros((5,)), chol_Q))
    transition_model = FunctionalModel(transition_function, MVNStandard(jnp.zeros((5,)), Q))

    sqrt_observation_model = FunctionalModel(observation_function, MVNSqrt(jnp.zeros((2,)), chol_R))
    observation_model = FunctionalModel(observation_function, MVNStandard(jnp.zeros((2,)), R))

    sqrt_iterated_res = iterated_smoothing(ys, chol_init, sqrt_transition_model, sqrt_observation_model,
                                           linearization_method, None, parallel,
                                           criterion=lambda i, *_: i < 100)

    iterated_res = iterated_smoothing(ys, init, transition_model, observation_model,
                                      linearization_method, None, parallel,
                                      criterion=lambda i, *_: i < 100)

    np.testing.assert_array_almost_equal(iterated_res.mean[1:], expected_mean, decimal=3)  # noqa
    np.testing.assert_array_almost_equal(iterated_res.cov[1:], expected_cov, decimal=3)  # noqa
    np.testing.assert_array_almost_equal(sqrt_iterated_res.mean[1:], expected_mean, decimal=3)  # noqa
    np.testing.assert_array_almost_equal(
        sqrt_iterated_res.chol[1:] @ np.transpose(sqrt_iterated_res.chol[1:], [0, 2, 1]),
        expected_cov, decimal=3)

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from parsmooth._base import MVNStandard, FunctionalModelX, MVNSqrt
from parsmooth.linearization import cubature, extended
from parsmooth.methods import iterated_smoothing
from tests.bearings.bearings_utils import make_parameters

LIST_LINEARIZATIONS = [extended, cubature]


@pytest.fixture(scope="session", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_disable_jit', False)
    jax.config.update("jax_debug_nans", False)


# @pytest.mark.skip("Skip on continuous integration")
@pytest.mark.parametrize("linearization_method", LIST_LINEARIZATIONS)
def test_bearings(linearization_method):
    s1 = jnp.array([-1.5, 0.5])  # First sensor location
    s2 = jnp.array([1., 1.])  # Second sensor location
    r = 0.5  # Observation noise (stddev)
    dt = 0.01  # discretization time step
    qc = 0.01  # discretization noise
    qw = 0.1  # discretization noise

    ys = np.load("./bearings/ys.npy")
    if linearization_method is extended:
        with np.load("./bearings//previous_results_new.npz") as loaded:
            expected_mean, expected_cov = loaded["expected_mean_extended"], loaded["expected_cov_extended"]
    elif linearization_method is cubature:
        with np.load("./bearings//previous_results_new.npz") as loaded:
            expected_mean, expected_cov = loaded["expected_mean_cubature"], loaded["expected_cov_cubature"]
    else:
        pytest.skip("We don't have regression data for this linearization")

    Q, R, observation_function, transition_function = make_parameters(qc, qw, r, dt, s1, s2)

    m0 = jnp.array([-1., -1., 0., 0., 0.])
    chol_P0 = P0 = jnp.eye(5)

    chol_Q = jnp.linalg.cholesky(Q)
    chol_R = jnp.linalg.cholesky(R)

    T = ys.shape[0]
    initial_states = MVNStandard(jnp.repeat(jnp.array([[-1., -1., 0., 0., 0.]]), T, axis=0),
                                 jnp.repeat(jnp.eye(5).reshape(1, 5, 5), T, axis=0))
    initial_states_sqrt = MVNSqrt(jnp.repeat(jnp.array([[-1., -1., 0., 0., 0.]]), T, axis=0),
                                  jnp.repeat(jnp.eye(5).reshape(1, 5, 5), T, axis=0))

    init = MVNStandard(m0, P0)
    chol_init = MVNSqrt(m0, chol_P0)

    sqrt_transition_model = FunctionalModelX(transition_function, MVNSqrt(jnp.zeros((5,)), chol_Q))
    transition_model = FunctionalModelX(transition_function, MVNStandard(jnp.zeros((5,)), Q))

    sqrt_observation_model = FunctionalModelX(observation_function, MVNSqrt(jnp.zeros((2,)), chol_R))
    observation_model = FunctionalModelX(observation_function, MVNStandard(jnp.zeros((2,)), R))

    iterated_res_par = iterated_smoothing(ys, init, transition_model, observation_model,
                                      linearization_method, initial_states, True,
                                      criterion=lambda i, *_: i < 101)

    iterated_res_seq = iterated_smoothing(ys, init, transition_model, observation_model,
                                      linearization_method, initial_states, False,
                                          criterion=lambda i, *_: i < 101)

    np.testing.assert_array_almost_equal(iterated_res_par.mean, expected_mean, decimal=7)  # noqa
    np.testing.assert_array_almost_equal(iterated_res_par.cov, expected_cov, decimal=7)  # noqa

    np.testing.assert_array_almost_equal(iterated_res_seq.mean, expected_mean, decimal=7)  # noqa
    np.testing.assert_array_almost_equal(iterated_res_seq.cov, expected_cov, decimal=7)  # noqa

    np.testing.assert_array_almost_equal(iterated_res_par.mean, iterated_res_seq.mean, decimal=10)  # noqa
    np.testing.assert_array_almost_equal(iterated_res_par.cov, iterated_res_seq.cov, decimal=10)  # noqa

    # sqrt_iterated_res_par = iterated_smoothing(ys, chol_init, sqrt_transition_model, sqrt_observation_model,
    #                                            linearization_method, initial_states_sqrt, True,
    #                                            criterion=lambda i, *_: i < 101)
    # sqrt_iterated_res_seq = iterated_smoothing(ys, chol_init, sqrt_transition_model, sqrt_observation_model,
    #                                            linearization_method, initial_states_sqrt, False,
    #                                            criterion=lambda i, *_: i < 101)

    # np.testing.assert_array_almost_equal(sqrt_iterated_res_par.mean, expected_mean, decimal=0)  # noqa
    # np.testing.assert_array_almost_equal(sqrt_iterated_res_par.chol @ np.transpose(sqrt_iterated_res_par.chol, [0, 2, 1]), expected_cov, decimal=0)

    # np.testing.assert_array_almost_equal(sqrt_iterated_res_seq.mean, expected_mean, decimal=0) # noqa
    # np.testing.assert_array_almost_equal(sqrt_iterated_res_seq.chol @ np.transpose(sqrt_iterated_res_seq.chol, [0, 2, 1]), expected_cov, decimal=0)

    # np.testing.assert_array_almost_equal(sqrt_iterated_res_par.mean, sqrt_iterated_res_seq.mean, decimal=4)  # noqa
    # np.testing.assert_array_almost_equal(sqrt_iterated_res_par.chol , sqrt_iterated_res_seq.chol, decimal=4)  # noqa


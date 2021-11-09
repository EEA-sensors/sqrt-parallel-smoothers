from functools import partial

import jax
import numpy as np
import pytest
from jax.scipy.linalg import solve

from parsmooth._base import FunctionalModel, MVNStandard, MVNSqrt
from parsmooth.linearization import cubature, extended
from parsmooth.sequential._filtering import filtering
from parsmooth.sequential._smoothing import _standard_smooth, _sqrt_smooth, smoothing
from tests._lgssm import get_data, transition_function as lgssm_f, observation_function as lgssm_h
from tests._test_utils import get_system

LIST_LINEARIZATIONS = [cubature, extended]


@pytest.fixture(scope="session", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update('jax_disable_jit', False)
    jax.config.update("jax_debug_nans", False)


@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
def test_smooth_one_standard_vs_sqrt(dim_x, seed):
    np.random.seed(seed)
    xf, chol_xf, F, Q, cholQ, b, _ = get_system(dim_x, dim_x)
    xs, chol_xs, *_ = get_system(dim_x, dim_x)

    next_x = _standard_smooth(F, Q, b, xf, xs)
    next_chol_x = _sqrt_smooth(F, cholQ, b, chol_xf, chol_xs)

    np.testing.assert_allclose(next_x.mean, next_chol_x.mean, atol=1e-5)
    np.testing.assert_allclose(next_x.cov, next_chol_x.chol @ next_chol_x.chol.T, atol=1e-3)


@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("sqrt", [True, False])
def test_smooth_one_value(dim_x, seed, sqrt):
    np.random.seed(seed)
    xf, chol_xf, F, Q, cholQ, b, _ = get_system(dim_x, dim_x)
    xs, chol_xs, *_ = get_system(dim_x, dim_x)
    if sqrt:
        next_x = _sqrt_smooth(F, cholQ, b, chol_xf, chol_xs)
    else:
        next_x = _standard_smooth(F, Q, b, xf, xs)

    if sqrt:
        cov = next_x.chol @ next_x.chol.T
    else:
        cov = next_x.cov

    m_ = F @ xf.mean + b
    P_ = F @ xf.cov @ F.T + Q
    G = xf.cov @ solve(P_.T, F, sym_pos=True).T
    ms = xf.mean + G @ (xs.mean - m_)
    Ps = xf.cov + G @ (xs.cov - P_) @ G.T

    np.testing.assert_allclose(next_x.mean, ms, atol=1e-5)
    np.testing.assert_allclose(cov, Ps, atol=1e-5)


@pytest.mark.parametrize("dim_x", [1, 3])
@pytest.mark.parametrize("dim_y", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("sqrt", [True, False])
def test_smooth_one_standard_vs_sqrt_no_noise(dim_x, dim_y, seed, sqrt):
    np.random.seed(seed)
    xf, chol_xf, F, Q, cholQ, b, _ = get_system(dim_x, dim_x)
    xs, chol_xs, *_ = get_system(dim_x, dim_x)
    Q = 0. * Q
    cholQ = 0. * cholQ

    if sqrt:
        next_x = _sqrt_smooth(F, cholQ, b, chol_xf, chol_xs)
    else:
        next_x = _standard_smooth(F, Q, b, xf, xs)

    if sqrt:
        cov = next_x.chol @ next_x.chol.T
    else:
        cov = next_x.cov

    m_ = F @ xf.mean + b
    P_ = F @ xf.cov @ F.T
    G = np.linalg.pinv(F)
    ms = xf.mean + G @ (xs.mean - m_)
    Ps = xf.cov + G @ (xs.cov - P_) @ G.T

    np.testing.assert_allclose(next_x.mean, ms, atol=1e-5)
    np.testing.assert_allclose(cov, Ps, atol=1e-5)


@pytest.mark.parametrize("dim_x", [1, 3])
@pytest.mark.parametrize("dim_y", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("sqrt", [True, False])
def test_smooth_one_standard_vs_sqrt_infinite_noise(dim_x, dim_y, seed, sqrt):
    np.random.seed(seed)
    xf, chol_xf, F, Q, cholQ, b, _ = get_system(dim_x, dim_x)
    xs, chol_xs, *_ = get_system(dim_x, dim_x)
    Q = 1e12 * Q
    cholQ = 1e6 * cholQ

    if sqrt:
        next_x = _sqrt_smooth(F, cholQ, b, chol_xf, chol_xs)
    else:
        next_x = _standard_smooth(F, Q, b, xf, xs)

    if sqrt:
        cov = next_x.chol @ next_x.chol.T
    else:
        cov = next_x.cov

    np.testing.assert_allclose(next_x.mean, xf.mean, atol=1e-5)
    np.testing.assert_allclose(cov, xf.cov, atol=1e-5)


@pytest.mark.parametrize("dim_x", [1, 3])
@pytest.mark.parametrize("dim_y", [2, 3])
@pytest.mark.parametrize("seed", [0, 42])
def test_all_smoothers_agree(dim_x, dim_y, seed):
    np.random.seed(seed)
    T = 5

    x0, chol_x0, F, Q, cholQ, b, _ = get_system(dim_x, dim_x)
    _, _, H, R, cholR, c, _ = get_system(dim_x, dim_y)

    true_states, observations = get_data(x0.mean, F, H, R, Q, b, c, T)
    sqrt_transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNSqrt(b, cholQ))
    sqrt_observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNSqrt(c, cholR))

    transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNStandard(b, Q))
    observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNStandard(c, R))

    res = []
    for method in LIST_LINEARIZATIONS:
        filtered_states = filtering(observations, x0, transition_model, observation_model, method,
                                    None)
        smoothed_states = smoothing(transition_model, filtered_states, method, None)

        sqrt_filtered_states = filtering(observations, chol_x0, sqrt_transition_model, sqrt_observation_model, method,
                                         None)
        sqrt_smoothed_states = smoothing(sqrt_transition_model, sqrt_filtered_states, method, None)
        res.append(smoothed_states)
        res.append(sqrt_smoothed_states)

    for res_1, res_2 in zip(res[:-1], res[1:]):
        np.testing.assert_array_almost_equal(res_1.mean, res_2.mean, decimal=3)


@pytest.mark.parametrize("dim_x", [1, 3])
@pytest.mark.parametrize("dim_y", [2, 3])
@pytest.mark.parametrize("seed", [0, 42])
def test_all_smoothers_with_nominal_traj(dim_x, dim_y, seed):
    np.random.seed(seed)
    T = 5
    m_nominal = np.random.randn(T + 1, dim_x)
    P_nominal = np.repeat(np.eye(dim_x, dim_x)[None, ...], T + 1, axis=0)
    cholP_nominal = P_nominal

    x_nominal = MVNStandard(m_nominal, P_nominal)
    sqrt_x_nominal = MVNSqrt(m_nominal, cholP_nominal)

    x0, chol_x0, F, Q, cholQ, b, _ = get_system(dim_x, dim_x)
    _, _, H, R, cholR, c, _ = get_system(dim_x, dim_y)

    true_states, observations = get_data(x0.mean, F, H, R, Q, b, c, T)
    sqrt_transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNSqrt(b, cholQ))
    sqrt_observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNSqrt(c, cholR))

    transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNStandard(b, Q))
    observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNStandard(c, R))
    for method in LIST_LINEARIZATIONS:
        filtered_states = filtering(observations, x0, transition_model, observation_model, method, None)
        smoothed_states_nominal = smoothing(transition_model, filtered_states, method, x_nominal)
        smoothed_states = smoothing(transition_model, filtered_states, method, None)

        sqrt_filtered_states = filtering(observations, chol_x0, sqrt_transition_model, sqrt_observation_model, method,
                                         None)
        sqrt_smoothed_states_nominal = smoothing(sqrt_transition_model, sqrt_filtered_states, method, sqrt_x_nominal)
        sqrt_smoothed_states = smoothing(sqrt_transition_model, sqrt_filtered_states, method, None)

        np.testing.assert_allclose(smoothed_states_nominal.mean, smoothed_states.mean, atol=1e-3)
        np.testing.assert_allclose(sqrt_smoothed_states.mean, sqrt_smoothed_states_nominal.mean, atol=1e-3)
        np.testing.assert_allclose(smoothed_states_nominal.cov, smoothed_states_nominal.cov, atol=1e-3)
        np.testing.assert_allclose(sqrt_smoothed_states.chol, sqrt_smoothed_states_nominal.chol, atol=1e-3)

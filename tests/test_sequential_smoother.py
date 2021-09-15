from functools import partial

import jax
import numpy as np
import pytest
from jax.scipy.linalg import solve

from parsmooth._base import FunctionalModel, MVNParams
from parsmooth.linearization import cubature, extended
from parsmooth.linearization._common import fix_mvn
from parsmooth.sequential._filter import filtering
from parsmooth.sequential._smoother import _standard_smooth, _sqrt_smooth, smoother
from tests._lgssm import get_data, transition_function as lgssm_f, observation_function as lgssm_h
from tests._test_utils import get_system

LIST_LINEARIZATIONS = [cubature, extended]


@pytest.fixture(scope="session", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_disable_jit', False)
    jax.config.update("jax_debug_nans", True)


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
@pytest.mark.parametrize("method", [_standard_smooth, _sqrt_smooth])
def test_smooth_one_value(dim_x, seed, method):
    np.random.seed(seed)
    xf, chol_xf, F, Q, cholQ, b, _ = get_system(dim_x, dim_x)
    xs, chol_xs, *_ = get_system(dim_x, dim_x)
    if method is _sqrt_smooth:
        next_x = method(F, cholQ, b, chol_xf, chol_xs)
    else:
        next_x = method(F, Q, b, xf, xs)
    next_x = fix_mvn(next_x)

    m_ = F @ xf.mean + b
    P_ = F @ xf.cov @ F.T + Q
    G = xf.cov @ solve(P_.T, F, sym_pos=True).T
    ms = xf.mean + G @ (xs.mean - m_)
    Ps = xf.cov + G @ (xs.cov - P_) @ G.T

    np.testing.assert_allclose(next_x.mean, ms, atol=1e-5)
    np.testing.assert_allclose(next_x.cov, Ps, atol=1e-5)


@pytest.mark.parametrize("dim_x", [1, 3])
@pytest.mark.parametrize("dim_y", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("method", [_standard_smooth, _sqrt_smooth])
def test_smooth_one_standard_vs_sqrt_no_noise(dim_x, dim_y, seed, method):
    np.random.seed(seed)
    xf, chol_xf, F, Q, cholQ, b, _ = get_system(dim_x, dim_x)
    xs, chol_xs, *_ = get_system(dim_x, dim_x)
    Q = 0. * Q
    cholQ = 0. * cholQ

    if method is _sqrt_smooth:
        next_x = method(F, cholQ, b, chol_xf, chol_xs)
    else:
        next_x = method(F, Q, b, xf, xs)
    next_x = fix_mvn(next_x)

    m_ = F @ xf.mean + b
    P_ = F @ xf.cov @ F.T
    G = np.linalg.pinv(F)
    ms = xf.mean + G @ (xs.mean - m_)
    Ps = xf.cov + G @ (xs.cov - P_) @ G.T

    np.testing.assert_allclose(next_x.mean, ms, atol=1e-5)
    np.testing.assert_allclose(next_x.cov, Ps, atol=1e-5)


@pytest.mark.parametrize("dim_x", [1, 3])
@pytest.mark.parametrize("dim_y", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("method", [_standard_smooth, _sqrt_smooth])
def test_smooth_one_standard_vs_sqrt_infinite_noise(dim_x, dim_y, seed, method):
    np.random.seed(seed)
    xf, chol_xf, F, Q, cholQ, b, _ = get_system(dim_x, dim_x)
    xs, chol_xs, *_ = get_system(dim_x, dim_x)
    Q = 1e12 * Q
    cholQ = 1e6 * cholQ

    if method is _sqrt_smooth:
        next_x = method(F, cholQ, b, chol_xf, chol_xs)
    else:
        next_x = method(F, Q, b, xf, xs)
    next_x = fix_mvn(next_x)
    xf = fix_mvn(xf)
    np.testing.assert_allclose(next_x.mean, xf.mean, atol=1e-5)
    np.testing.assert_allclose(next_x.cov, xf.cov, atol=1e-5)


@pytest.mark.parametrize("dim_x", [1, 3])
@pytest.mark.parametrize("dim_y", [2, 3])
@pytest.mark.parametrize("seed", [0, 42])
def test_all_smoothers_agree(dim_x, dim_y, seed):
    np.random.seed(seed)
    T = 5

    x0, chol_x0, F, Q, cholQ, b, _ = get_system(dim_x, dim_x)
    x0 = MVNParams(x0.mean, x0.cov, chol_x0.chol)
    _, _, H, R, cholR, c, _ = get_system(dim_x, dim_y)

    true_states, observations = get_data(x0.mean, F, H, R, Q, b, c, T)
    transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNParams(b, Q, cholQ))
    observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNParams(c, R, cholR))

    res = []
    for method in LIST_LINEARIZATIONS:
        for sqrt in [True, False]:
            filtered_states = filtering(observations, x0, transition_model, observation_model, method,
                                        sqrt, None)
            smoothed_states = smoother(transition_model, filtered_states, None, method, sqrt)
            res.append(smoothed_states)

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

    x_nominal = MVNParams(m_nominal, P_nominal, cholP_nominal)

    x0, chol_x0, F, Q, cholQ, b, _ = get_system(dim_x, dim_x)
    x0 = MVNParams(x0.mean, x0.cov, chol_x0.chol)
    _, _, H, R, cholR, c, _ = get_system(dim_x, dim_y)

    true_states, observations = get_data(x0.mean, F, H, R, Q, b, c, T)
    transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNParams(b, Q, cholQ))
    observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNParams(c, R, cholR))

    for method in LIST_LINEARIZATIONS:
        for sqrt in [True, False]:
            filtered_states = filtering(observations, x0, transition_model, observation_model, method,
                                        sqrt, None)
            smoothed_states = smoother(transition_model, filtered_states, x_nominal, method, sqrt)
            smoothed_states_nominal = smoother(transition_model, filtered_states, None, method, sqrt)
            np.testing.assert_allclose(smoothed_states_nominal.mean, smoothed_states.mean, atol=1e-3)

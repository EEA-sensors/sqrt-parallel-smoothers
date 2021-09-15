from functools import partial

import jax
import numpy as np
import pytest
from jax.scipy.linalg import solve

from parsmooth._base import FunctionalModel, MVNParams
from parsmooth.linearization import cubature, extended
from parsmooth.linearization._common import fix_mvn
from parsmooth.sequential._filter import _sqrt_predict, _standard_predict, _sqrt_update, _standard_update, filtering
from tests._lgssm import get_data, transition_function as lgssm_f, observation_function as lgssm_h
from tests._test_utils import get_system

LIST_LINEARIZATIONS = [cubature, extended]


@pytest.fixture(scope="session", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_disable_jit', False)
    jax.config.update("jax_debug_nans", False)


@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
def test_predict_standard_vs_sqrt(dim_x, seed):
    np.random.seed(seed)
    x, chol_x, F, Q, cholQ, b, _ = get_system(dim_x, dim_x)

    next_chol_x = _sqrt_predict(F, cholQ, b, chol_x)
    next_x = _standard_predict(F, Q, b, x)

    np.testing.assert_allclose(next_x.mean, next_chol_x.mean, atol=1e-5)
    np.testing.assert_allclose(next_x.cov, next_chol_x.chol @ next_chol_x.chol.T, atol=1e-5)


@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("method", [_sqrt_predict, _standard_predict])
def test_predict_value(dim_x, seed, method):
    np.random.seed(seed)
    x, chol_x, F, Q, cholQ, b, _ = get_system(dim_x, dim_x)

    x = fix_mvn(x)
    if method is _sqrt_predict:
        next_x = method(F, cholQ, b, x)
    else:
        next_x = method(F, Q, b, x)
    next_x = fix_mvn(next_x)

    np.testing.assert_allclose(next_x.mean, F @ x.mean + b, atol=1e-5)
    np.testing.assert_allclose(next_x.cov, Q + F @ x.cov @ F.T, atol=1e-5)


@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("dim_y", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("method", [_sqrt_update, _standard_update])
def test_update_value(dim_x, dim_y, seed, method):
    np.random.seed(seed)
    x, _, H, R, cholR, c, y = get_system(dim_x, dim_y)

    x = fix_mvn(x)

    if method is _sqrt_update:
        next_x = method(H, cholR, c, x, y)
    else:
        next_x = method(H, R, c, x, y)
    next_x = fix_mvn(next_x)

    res = y - H @ x.mean - c
    S = H @ x.cov @ H.T + R
    K = x.cov @ solve(S, H, sym_pos=True).T
    np.testing.assert_allclose(next_x.mean, x.mean + K @ res, atol=1e-1)
    np.testing.assert_allclose(next_x.cov, x.cov - K @ H @ x.cov, atol=1e-5)


@pytest.mark.parametrize("dim_x", [1, 3])
@pytest.mark.parametrize("dim_y", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
def test_update_standard_vs_sqrt(dim_x, dim_y, seed):
    np.random.seed(seed)
    x, chol_x, H, R, cholR, c, y = get_system(dim_x, dim_y)

    x = _standard_update(H, R, c, x, y)
    chol_x = _sqrt_update(H, cholR, c, chol_x, y)

    np.testing.assert_allclose(x.cov, chol_x.chol @ chol_x.chol.T, atol=1e-5)
    np.testing.assert_allclose(x.mean, chol_x.mean, atol=1e-5)


@pytest.mark.parametrize("dim_x", [1, 3])
@pytest.mark.parametrize("dim_y", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
def test_update_standard_vs_sqrt_no_noise(dim_x, dim_y, seed):
    np.random.seed(seed)
    x, chol_x, H, R, cholR, c, y = get_system(dim_x, dim_y)
    R *= 1e9
    cholR *= 1e9
    next_x = _standard_update(H, R, c, x, y)
    next_chol_x = _sqrt_update(H, cholR, c, chol_x, y)

    np.testing.assert_allclose(x.cov, next_x.cov, atol=1e-3)
    np.testing.assert_allclose(x.mean, next_x.mean, atol=1e-3)
    np.testing.assert_allclose(chol_x.chol @ chol_x.chol.T, next_chol_x.chol @ next_chol_x.chol.T, atol=1e-3)
    np.testing.assert_allclose(x.mean, next_x.mean, atol=1e-3)


@pytest.mark.parametrize("dim_x", [1, 3])
@pytest.mark.parametrize("dim_y", [2, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("sqrt", [False, True])
@pytest.mark.parametrize("linearization_method", LIST_LINEARIZATIONS)
def test_filter_no_info(dim_x, dim_y, seed, sqrt, linearization_method):
    np.random.seed(seed)
    T = 3

    x0, chol_x0, F, Q, cholQ, b, _ = get_system(dim_x, dim_x)
    x0 = MVNParams(x0.mean, x0.cov, chol_x0.chol)
    _, _, H, R, cholR, c, _ = get_system(dim_x, dim_y)
    R = 1e12 * R
    cholR = 1e6 * cholR
    true_states, observations = get_data(x0.mean, F, H, R, Q, b, c, T)
    transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNParams(b, Q, cholQ))
    observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNParams(c, R, cholR))
    fun = lambda x: F @ x + b
    expected_mean = [x0.mean]

    for t in range(T):
        expected_mean.append(fun(expected_mean[-1]))
    filtered_states = filtering(observations, x0, transition_model, observation_model, linearization_method, sqrt, None)
    np.testing.assert_allclose(filtered_states.mean, expected_mean, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dim", [1, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("sqrt", [False, True])
@pytest.mark.parametrize("linearization_method", LIST_LINEARIZATIONS)
def test_filter_infinite_info(dim, seed, sqrt, linearization_method):
    np.random.seed(seed)
    T = 3

    x0, chol_x0, _, Q, cholQ, b, _ = get_system(dim, dim)
    F = np.eye(dim)
    x0 = MVNParams(x0.mean, x0.cov, chol_x0.chol)
    _, _, _, R, cholR, c, _ = get_system(dim, dim)
    H = np.eye(dim)

    R = 1e-6 * R
    cholR = 1e-3 * cholR
    true_states, observations = get_data(x0.mean, F, H, R, Q, b, c, T, chol_R=cholR)
    transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNParams(b, Q, cholQ))
    observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNParams(c, R, cholR))
    expected_mean = np.stack([y - c for y in observations], axis=0)

    filtered_states = filtering(observations, x0, transition_model, observation_model, linearization_method, sqrt, None)
    np.testing.assert_allclose(filtered_states.mean[1:], expected_mean, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dim_x", [1, 3])
@pytest.mark.parametrize("dim_y", [2, 3])
@pytest.mark.parametrize("seed", [0, 42])
def test_all_filters_agree(dim_x, dim_y, seed):
    np.random.seed(seed)
    T = 25

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
            res.append(filtered_states)

    for res_1, res_2 in zip(res[:-1], res[1:]):
        np.testing.assert_array_almost_equal(res_1.mean, res_2.mean, decimal=3)


@pytest.mark.parametrize("dim_x", [1, 3])
@pytest.mark.parametrize("dim_y", [2, 3])
@pytest.mark.parametrize("seed", [0, 42])
def test_all_filters_with_nominal_traj(dim_x, dim_y, seed):
    np.random.seed(seed)
    T = 25
    m_nominal = np.random.randn(T+1, dim_x)
    P_nominal = np.repeat(np.eye(dim_x, dim_x)[None, ...], T+1, axis=0)
    cholP_nominal = P_nominal

    x_nominal = MVNParams(m_nominal, P_nominal, cholP_nominal)

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
            filtered_states_nominal = filtering(observations, x0, transition_model, observation_model, method,
                                                sqrt, x_nominal)
            np.testing.assert_allclose(filtered_states_nominal.mean, filtered_states.mean, atol=1e-3)
            res.append(filtered_states)


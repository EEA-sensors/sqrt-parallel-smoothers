from functools import partial

import jax
import numpy as np
import pytest
from jax.scipy.linalg import solve

from parsmooth._base import FunctionalModel, MVNStandard, MVNSqrt
from parsmooth._utils import mvn_loglikelihood
from parsmooth.linearization import cubature, extended
from parsmooth.sequential._filtering import _sqrt_predict, _standard_predict, _sqrt_update, _standard_update, filtering
from tests._lgssm import get_data, transition_function as lgssm_f, observation_function as lgssm_h
from tests._test_utils import get_system

LIST_LINEARIZATIONS = [cubature]


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
@pytest.mark.parametrize("sqrt", [True, False])
def test_predict_value(dim_x, seed, sqrt):
    np.random.seed(seed)
    x, chol_x, F, Q, cholQ, b, _ = get_system(dim_x, dim_x)

    if sqrt:
        next_x = _sqrt_predict(F, cholQ, b, chol_x)
    else:
        next_x = _standard_predict(F, Q, b, x)

    if sqrt:
        cov = next_x.chol @ next_x.chol.T
    else:
        cov = next_x.cov

    np.testing.assert_allclose(next_x.mean, F @ x.mean + b, atol=1e-5)
    np.testing.assert_allclose(cov, Q + F @ x.cov @ F.T, atol=1e-5)


@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("dim_y", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("sqrt", [True, False])
def test_update_value(dim_x, dim_y, seed, sqrt):
    np.random.seed(seed)
    x, chol_x, H, R, cholR, c, y = get_system(dim_x, dim_y)

    if sqrt:
        next_x, ell = _sqrt_update(H, cholR, c, chol_x, y)
    else:
        next_x, ell = _standard_update(H, R, c, x, y)

    if sqrt:
        cov = next_x.chol @ next_x.chol.T
    else:
        cov = next_x.cov

    res = y - H @ x.mean - c
    S = H @ x.cov @ H.T + R
    K = x.cov @ solve(S, H, sym_pos=True).T
    np.testing.assert_allclose(next_x.mean, x.mean + K @ res, atol=1e-1)
    np.testing.assert_allclose(cov, x.cov - K @ H @ x.cov, atol=1e-5)

    expected_ell = mvn_loglikelihood(res, np.linalg.cholesky(S))
    np.testing.assert_allclose(expected_ell, ell, atol=1e-5)


@pytest.mark.parametrize("dim_x", [1, 3])
@pytest.mark.parametrize("dim_y", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
def test_update_standard_vs_sqrt(dim_x, dim_y, seed):
    np.random.seed(seed)
    x, chol_x, H, R, cholR, c, y = get_system(dim_x, dim_y)

    x, ell = _standard_update(H, R, c, x, y)
    chol_x, chol_ell = _sqrt_update(H, cholR, c, chol_x, y)

    np.testing.assert_allclose(x.cov, chol_x.chol @ chol_x.chol.T, atol=1e-5)
    np.testing.assert_allclose(x.mean, chol_x.mean, atol=1e-5)
    np.testing.assert_allclose(ell, chol_ell, atol=1e-5)


@pytest.mark.parametrize("dim_x", [1, 3])
@pytest.mark.parametrize("dim_y", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
def test_update_standard_vs_sqrt_no_info(dim_x, dim_y, seed):
    np.random.seed(seed)
    x, chol_x, H, R, cholR, c, y = get_system(dim_x, dim_y)
    R *= 1e8
    cholR *= 1e4
    next_x, ell = _standard_update(H, R, c, x, y)
    next_chol_x, chol_ell = _sqrt_update(H, cholR, c, chol_x, y)

    np.testing.assert_allclose(ell, chol_ell, atol=1e-3)
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
    if sqrt:
        x0 = chol_x0
    _, _, H, R, cholR, c, _ = get_system(dim_x, dim_y)

    R = 1e12 * R
    cholR = 1e6 * cholR
    true_states, observations = get_data(x0.mean, F, H, R, Q, b, c, T)

    if sqrt:
        transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNSqrt(b, cholQ))
        observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNSqrt(c, cholR))
    else:
        transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNStandard(b, Q))
        observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNStandard(c, R))

    fun = lambda x: F @ x + b
    expected_mean = [x0.mean]

    for t in range(T):
        expected_mean.append(fun(expected_mean[-1]))

    filtered_states = filtering(observations, x0, transition_model, observation_model, linearization_method, None)
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
    if sqrt:
        x0 = chol_x0
    _, _, _, R, cholR, c, _ = get_system(dim, dim)
    H = np.eye(dim)

    R = 1e-6 * R
    cholR = 1e-3 * cholR
    true_states, observations = get_data(x0.mean, F, H, R, Q, b, c, T, chol_R=cholR)

    if sqrt:
        transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNSqrt(b, cholQ))
        observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNSqrt(c, cholR))
    else:
        transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNStandard(b, Q))
        observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNStandard(c, R))

    expected_mean = np.stack([y - c for y in observations], axis=0)

    filtered_states = filtering(observations, x0, transition_model, observation_model, linearization_method, None)
    np.testing.assert_allclose(filtered_states.mean[1:], expected_mean, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dim_x", [1, 3])
@pytest.mark.parametrize("dim_y", [2, 3])
@pytest.mark.parametrize("seed", [0, 42])
def test_all_filters_agree(dim_x, dim_y, seed):
    np.random.seed(seed)
    T = 4

    x0, chol_x0, F, Q, cholQ, b, _ = get_system(dim_x, dim_x)
    _, _, H, R, cholR, c, _ = get_system(dim_x, dim_y)

    true_states, observations = get_data(x0.mean, F, H, R, Q, b, c, T)

    sqrt_transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNSqrt(b, cholQ))
    sqrt_observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNSqrt(c, cholR))

    transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNStandard(b, Q))
    observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNStandard(c, R))

    res = []
    for method in LIST_LINEARIZATIONS:
        filtered_states = filtering(observations, x0, transition_model, observation_model, method, None)
        sqrt_filtered_states = filtering(observations, chol_x0, sqrt_transition_model, sqrt_observation_model, method,
                                         None)
        res.append(filtered_states)
        res.append(sqrt_filtered_states)

    for res_1, res_2 in zip(res[:-1], res[1:]):
        np.testing.assert_array_almost_equal(res_1.mean, res_2.mean, decimal=3)


@pytest.mark.parametrize("dim_x", [1, 3])
@pytest.mark.parametrize("dim_y", [2, 3])
@pytest.mark.parametrize("seed", [0, 42])
def test_all_filters_with_nominal_traj(dim_x, dim_y, seed):
    np.random.seed(seed)
    T = 5
    m_nominal = np.random.randn(T + 1, dim_x)
    P_nominal = np.repeat(np.eye(dim_x, dim_x)[None, ...], T + 1, axis=0)
    cholP_nominal = P_nominal

    x_nominal = MVNStandard(m_nominal, P_nominal)
    x_nominal_sqrt = MVNSqrt(m_nominal, cholP_nominal)

    x0, chol_x0, F, Q, cholQ, b, _ = get_system(dim_x, dim_x)
    _, _, H, R, cholR, c, _ = get_system(dim_x, dim_y)

    true_states, observations = get_data(x0.mean, F, H, R, Q, b, c, T)

    sqrt_transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNSqrt(b, cholQ))
    sqrt_observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNSqrt(c, cholR))

    transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNStandard(b, Q))
    observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNStandard(c, R))

    for method in LIST_LINEARIZATIONS:
        filtered_states = filtering(observations, x0, transition_model, observation_model, method,
                                    None)
        filtered_states_nominal = filtering(observations, x0, transition_model, observation_model, method,
                                            x_nominal)
        sqrt_filtered_states = filtering(observations, chol_x0, sqrt_transition_model, sqrt_observation_model, method,
                                         None)
        sqrt_filtered_states_nominal = filtering(observations, chol_x0, sqrt_transition_model, sqrt_observation_model,
                                                 method, x_nominal_sqrt)

        np.testing.assert_allclose(filtered_states_nominal.mean, filtered_states.mean, atol=1e-3)
        np.testing.assert_allclose(filtered_states_nominal.mean, sqrt_filtered_states_nominal.mean, atol=1e-3)

        np.testing.assert_allclose(sqrt_filtered_states.mean, sqrt_filtered_states_nominal.mean, atol=1e-3)

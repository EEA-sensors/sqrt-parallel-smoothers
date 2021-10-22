from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, solve_triangular

from parsmooth._base import MVNStandard, FunctionalModel, MVNSqrt, are_inputs_compatible, ConditionalMomentsModel
from parsmooth._utils import tria, none_or_shift, none_or_concat, mvn_loglikelihood


def filtering(observations: jnp.ndarray,
              x0: Union[MVNSqrt, MVNStandard],
              transition_model: Union[FunctionalModel, ConditionalMomentsModel],
              observation_model: Union[FunctionalModel, ConditionalMomentsModel],
              linearization_method: Callable,
              nominal_trajectory: Optional[Union[MVNSqrt, MVNStandard]] = None,
              return_loglikelihood: bool = False):
    if nominal_trajectory is not None:
        are_inputs_compatible(x0, nominal_trajectory)

    def predict(F_x, cov_or_chol, b, x):
        if isinstance(x, MVNSqrt):
            return _sqrt_predict(F_x, cov_or_chol, b, x)
        return _standard_predict(F_x, cov_or_chol, b, x)

    def update(H_x, cov_or_chol, c, x, y):
        if isinstance(x, MVNSqrt):
            return _sqrt_update(H_x, cov_or_chol, c, x, y)
        return _standard_update(H_x, cov_or_chol, c, x, y)

    def body(carry, inp):
        x, ell = carry
        y, predict_ref, update_ref = inp

        if predict_ref is None:
            predict_ref = x
        F_x, cov_or_chol_Q, b = linearization_method(transition_model, predict_ref)
        x = predict(F_x, cov_or_chol_Q, b, x)
        if update_ref is None:
            update_ref = x
        H_x, cov_or_chol_R, c = linearization_method(observation_model, update_ref)
        x, ell_inc = update(H_x, cov_or_chol_R, c, x, y)
        return (x, ell + ell_inc), x

    predict_traj = none_or_shift(nominal_trajectory, -1)
    update_traj = none_or_shift(nominal_trajectory, 1)

    (_, ell), xs = jax.lax.scan(body, (x0, 0.), (observations, predict_traj, update_traj))
    xs = none_or_concat(xs, x0, 1)
    if return_loglikelihood:
        return xs, ell
    else:
        return xs


def _standard_predict(F, Q, b, x):
    m, P = x

    m = F @ m + b
    P = Q + F @ P @ F.T

    return MVNStandard(m, P)


def _standard_update(H, R, c, x, y):
    m, P = x

    y_hat = H @ m + c
    y_diff = y - y_hat
    S = R + H @ P @ H.T
    chol_S = jnp.linalg.cholesky(S)
    G = P @ cho_solve((chol_S, True), H).T

    m = m + G @ y_diff
    P = P - G @ S @ G.T
    ell = mvn_loglikelihood(y_diff, chol_S)
    return MVNStandard(m, P), ell


def _sqrt_predict(F, cholQ, b, x):
    m, cholP = x

    m = F @ m + b
    cholP = tria(jnp.concatenate([F @ cholP, cholQ], axis=1))

    return MVNSqrt(m, cholP)


def _sqrt_update(H, cholR, c, x, y):
    m, cholP = x
    nx = m.shape[0]
    ny = y.shape[0]

    y_hat = H @ m + c
    y_diff = y - y_hat

    M = jnp.block([[H @ cholP, cholR],
                   [cholP, jnp.zeros_like(cholP, shape=(nx, ny))]])
    chol_S = tria(M)

    cholP = chol_S[ny:, ny:]

    G = chol_S[ny:, :ny]
    I = chol_S[:ny, :ny]

    m = m + G @ solve_triangular(I, y_diff, lower=True)
    ell = mvn_loglikelihood(y_diff, I)
    return MVNSqrt(m, cholP), ell

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve, solve_triangular

from parsmooth._base import MVNParams, FunctionalModel
from parsmooth._utils import cholesky_update_many, tria, none_or_shift, none_or_concat


def filtering(observations: jnp.ndarray,
              x0: MVNParams,
              transition_model: FunctionalModel,
              observation_model: FunctionalModel,
              linearization_method: Callable,
              sqrt: bool,
              nominal_trajectory: Optional[jnp.ndarray] = None):
    f, mvn_Q = transition_model
    h, mvn_R = observation_model

    if sqrt:
        x0 = MVNParams(x0.mean, None, x0.chol)
    else:
        x0 = MVNParams(x0.mean, x0.cov)

    def predict(F_x, cov_or_chol, b, x):
        if sqrt:
            return _sqrt_predict(F_x, cov_or_chol, b, x)
        return _standard_predict(F_x, cov_or_chol, b, x)

    def update(H_x, cov_or_chol, c, x, y):
        if sqrt:
            return _sqrt_update(H_x, cov_or_chol, c, x, y)
        return _standard_update(H_x, cov_or_chol, c, x, y)

    def body(x, inp):
        y, predict_ref, update_ref = inp

        if predict_ref is None:
            predict_ref = x
        F_x, cov_or_chol_Q, b = linearization_method(f, predict_ref, mvn_Q, sqrt)
        x = predict(F_x, cov_or_chol_Q, b, x)

        if update_ref is None:
            update_ref = x
        H_x, cov_or_chol_R, c = linearization_method(h, update_ref, mvn_R, sqrt)
        x = update(F_x, cov_or_chol_R, b, x, y)
        return x, x

    predict_traj = none_or_shift(nominal_trajectory, -1)
    update_traj = none_or_shift(nominal_trajectory, 1)

    _, xs = jax.lax.scan(body, x0, (observations, predict_traj, update_traj))
    xs = MVNParams(*(none_or_concat(i, j) for i, j in zip(x0, xs)))
    return xs


def _standard_predict(F, Q, b, x):
    m, P, _ = x

    m = F @ m + b
    P = P + F @ Q @ F.T

    return MVNParams(m, P)


def _standard_update(H, R, c, x, y):
    m, P, _ = x

    y_hat = H @ m + c
    y_diff = y - y_hat
    S = R + H @ P @ H.T

    G = P @ solve(S, H, sym_pos=True).T

    m = m + G @ y_diff
    P = P - G @ S @ G.T
    return MVNParams(m, P)


def _sqrt_predict(F, cholQ, b, x):
    m, _, cholP = x

    m = F @ m + b
    cholP = cholesky_update_many(cholP, (F @ cholQ).T, 1.)

    return MVNParams(m, None, cholP)


def _sqrt_update(H, cholR, c, x, y):
    m, _, cholP = x
    nx = m.shape[0]
    ny = y.shape[0]

    y_hat = H @ m + c
    y_diff = y - y_hat

    M = jnp.block([[cholR, H @ cholP],
                   [jnp.zeros_like(cholP, shape=(nx, ny)), cholP]])
    S = tria(M)

    cholP = S[ny:, ny:]

    G = S[ny:, :ny]
    I = S[:ny, :ny]

    m = m + G @ solve_triangular(I, y_diff, lower=True)

    return MVNParams(m, None, cholP)

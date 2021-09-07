from typing import Callable, Optional

import jax.numpy as jnp
from jax.scipy.linalg import solve, solve_triangular

from parsmooth._base import MVNParams, FunctionalModel
from parsmooth._math_utils import cholesky_update_many, tria


def filtering(observations: jnp.ndarray, transition_model: FunctionalModel, observation_model: FunctionalModel,
              linearization_method: Callable, sqrt: bool, nominal_trajectory: Optional[jnp.ndarray] = None):
    predict = _standard_predict if not sqrt else _sqrt_predict
    update = _standard_update if not sqrt else _sqrt_update

    f, mvn_Q = transition_model
    h, mvn_R = observation_model

    def predict(F_x, F_q, mvn_Q, b, x):
        if sqrt:
            return _sqrt_predict(F_x, mvn_Q.chol, b, x)
        return _standard_predict(F_x, mvn_Q.cov, b, x)

    def body(x, inp):
        y, predict_ref, update_ref = inp
        if predict_ref is None:
            predict_ref = x

        F_x, F_q, b, _ = linearization_method(f, nominal_trajectory, mvn_Q, sqrt)

        if sqrt:
            x = _sqrt_predict(F_x, mvn_Q.chol, b, x)
        else:
            x = _standard_predict(F_x, mvn_Q.cov, b, x)


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

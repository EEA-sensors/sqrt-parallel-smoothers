import jax.numpy as jnp
from jax.scipy.linalg import solve, solve_triangular

from parsmooth._base import MVNParams
from parsmooth._math_utils import cholesky_update_many, tria


def filter(observations, transition_function, observation_function, linearization_method, sqrt, nominal_trajectory):
    pass


def _standard_filter():
    pass


def _sqrt_filter():
    pass


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

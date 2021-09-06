from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import block_diag
from jax.scipy.linalg import solve_triangular

from parsmooth._base import MVNParams
from parsmooth._math_utils import cholesky_update_many
from parsmooth.linearization._common import fix_mvn


class SigmaPoints(NamedTuple):
    points: np.ndarray
    wm: np.ndarray
    wc: np.ndarray


def _cov(wc, x_pts, x_mean, y_points, y_mean):
    one = (x_pts - x_mean[None, :]).T * wc[None, :]
    two = y_points - y_mean[None, :]
    return jnp.dot(one, two)


def linearize_callable(f, x, q, get_sigma_points, sqrt):
    x = fix_mvn(x)
    q = fix_mvn(q)
    m_x, chol_x, _ = x
    m_q, chol_q, _ = q
    dim_x = m_x.shape[0]
    xq = _concatenate_mvns(x, q)

    xq_pts, u_pts = get_sigma_points(xq)

    x_pts, q_pts = jnp.split(xq_pts.points, [dim_x], axis=1)
    ux_pts, uq_pts = jnp.split(u_pts, [dim_x], axis=1)

    f_pts = jax.vmap(f)(x_pts, q_pts)
    f_mean = jnp.dot(xq_pts.wm, f_pts)

    Psi_x = _cov(xq_pts.wc, ux_pts, jnp.zeros_like(m_x), f_pts, f_mean)
    Psi_q = _cov(xq_pts.wc, uq_pts, jnp.zeros_like(m_q), f_pts, f_mean)

    F_x = solve_triangular(x.chol, Psi_x, trans="T", lower=True).T
    F_q = solve_triangular(q.chol, Psi_q, trans="T", lower=True).T

    if sqrt:
        update_vectors = xq_pts.wc ** 0.5 * (f_pts - f_mean[None, :])
        chol_L = cholesky_update_many(F_x @ chol_x, update_vectors.T, -1.)
        return F_x, F_q, f_mean, chol_L

    Phi = _cov(xq_pts.wc, f_pts, f_mean, f_pts, f_mean)
    L = Phi - F_x @ x.cov @ F_x.T - F_q @ q.cov @ F_q.T
    return F_x, F_q, f_mean, L


def _concatenate_mvns(x, q):
    # This code implicitly assumes that X and Q are independent multivariate Gaussians.
    m_x, cov_x, chol_x = x
    m_q, cov_q, chol_q = q

    if chol_x is None:
        chol_x = jnp.linalg.cholesky(cov_x)

    if chol_q is None:
        chol_q = jnp.linalg.cholesky(cov_q)

    m_xq = jnp.concatenate([m_x, m_q])
    chol_xq = block_diag(chol_x, chol_q)
    return MVNParams(m_xq, None, chol_xq)

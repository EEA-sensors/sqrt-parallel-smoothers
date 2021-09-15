from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_solve, block_diag

from parsmooth._base import MVNSqrt, are_inputs_compatible
from parsmooth._utils import cholesky_update_many, tria
from parsmooth.linearization._common import get_mvnsqrt


class SigmaPoints(NamedTuple):
    points: np.ndarray
    wm: np.ndarray
    wc: np.ndarray


def _cov(wc, x_pts, x_mean, y_points, y_mean):
    one = (x_pts - x_mean[None, :]).T * wc[None, :]
    two = y_points - y_mean[None, :]
    return jnp.dot(one, two)


def linearize_callable(f, x, q, get_sigma_points):
    are_inputs_compatible(x, q)

    F_x, F_q, xq_pts, f_pts, m_f = _linearize_callable_common(f, x, q, get_sigma_points)
    m_q, _ = q
    if isinstance(x, MVNSqrt):
        m_x, chol_x = x
        sqrt_Phi = jnp.sqrt(xq_pts.wc[:, None]) * (f_pts - m_f[None, :])
        sqrt_Phi = tria(sqrt_Phi.T)
        chol_L = cholesky_update_many(sqrt_Phi, (F_x @ chol_x).T, -1.)
        return F_x, chol_L, m_f - F_x @ m_x
    m_x, cov_x = x
    Phi = _cov(xq_pts.wc, f_pts, m_f, f_pts, m_f)
    L = Phi - F_x @ cov_x @ F_x.T

    return F_x, L, m_f - F_x @ m_x


def _linearize_callable_common(f, x, q, get_sigma_points):
    x = get_mvnsqrt(x)
    q = get_mvnsqrt(q)
    m_x, chol_x = x
    m_q, chol_q = q
    dim_x = m_x.shape[0]
    xq = _concatenate_mvns(x, q)

    xq_pts = get_sigma_points(xq)

    x_pts, q_pts = jnp.split(xq_pts.points, [dim_x], axis=1)

    f_pts = jax.vmap(f)(x_pts, q_pts)
    m_f = jnp.dot(xq_pts.wm, f_pts)

    Psi_x = _cov(xq_pts.wc, x_pts, m_x, f_pts, m_f)
    Psi_q = _cov(xq_pts.wc, q_pts, m_q, f_pts, m_f)

    F_x = cho_solve((chol_x, True), Psi_x).T
    F_q = cho_solve((chol_q, True), Psi_q).T
    return F_x, F_q, xq_pts, f_pts, m_f


def _concatenate_mvns(x, q):
    # This code implicitly assumes that X and Q are independent multivariate Gaussians.
    m_x, chol_x = x
    m_q, chol_q = q
    m_xq = jnp.concatenate([m_x, m_q])
    chol_xq = block_diag(chol_x, chol_q)
    return MVNSqrt(m_xq, chol_xq)

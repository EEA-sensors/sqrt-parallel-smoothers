from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.host_callback import id_print
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


def linearize_conditional(c_m, c_cov_or_chol, x, get_sigma_points):
    F_x, x_pts, f_pts, m_f, v_f = linearize_conditional_common(c_m, c_cov_or_chol, x, get_sigma_points)
    if isinstance(x, MVNSqrt):
        m_x, chol_x = x
        
        sqrt_Phi = jnp.sqrt(x_pts.wc[:, None]) * (f_pts - m_f[None, :])
        sqrt_Phi = tria(sqrt_Phi.T)
        
        chol_L = cholesky_update_many(sqrt_Phi, (F_x @ chol_x).T, -1.)
        chol_pts = jax.vmap(c_cov_or_chol)(x_pts.points)
        
        chol_f = jnp.sum(x_pts.wc[:, None, None] * chol_pts, 0)
        
        L = tria(jnp.concatenate([chol_L, chol_f], axis=1))
        return F_x, L, m_f - F_x @ m_x

    m_x, cov_x = x
    Phi = _cov(x_pts.wc, f_pts, m_f, f_pts, m_f)
    L = Phi - F_x @ cov_x @ F_x.T + v_f
    return F_x, L, m_f - F_x @ m_x


def linearize_conditional_common(c_m, c_cov_or_chol, x, get_sigma_points):
    x = get_mvnsqrt(x)
    m_x, chol_x = x
    x_pts = get_sigma_points(x)

    f_pts = jax.vmap(c_m)(x_pts.points)
    V_pts = jax.vmap(c_cov_or_chol)(x_pts.points)
    
    m_f = jnp.dot(x_pts.wm, f_pts)
    v_f = jnp.sum(x_pts.wc[:, None, None] * V_pts, 0)
    
    Psi_x = _cov(x_pts.wc, x_pts.points, m_x, f_pts, m_f)
    F_x = cho_solve((chol_x, True), Psi_x).T
    return F_x, x_pts, f_pts, m_f, v_f


def linearize_functional(f, x, q, get_sigma_points):
    are_inputs_compatible(x, q)

    F_x, x_pts, f_pts, m_f = _linearize_functional_common(f, x, get_sigma_points)
    if isinstance(x, MVNSqrt):
        m_x, chol_x = x
        m_q, chol_q = q
        sqrt_Phi = jnp.sqrt(x_pts.wc[:, None]) * (f_pts - m_f[None, :])
        n_sigma_points, dim_out = sqrt_Phi.shape
        if n_sigma_points >= dim_out:
            sqrt_Phi = tria(sqrt_Phi.T)
        else:
            sqrt_Phi = jnp.concatenate([sqrt_Phi.T, jnp.zeros((dim_out, dim_out - n_sigma_points))], axis=1)
        chol_L = cholesky_update_many(sqrt_Phi, (F_x @ chol_x).T, -1.)
        chol_L = tria(jnp.concatenate([chol_L, chol_q], axis=1))
        return F_x, chol_L, m_f - F_x @ m_x + m_q
    m_x, cov_x = x
    m_q, cov_q = q
    Phi = _cov(x_pts.wc, f_pts, m_f, f_pts, m_f)
    L = Phi - F_x @ cov_x @ F_x.T + cov_q

    return F_x, 0.5 * (L + L.T), m_f - F_x @ m_x + m_q


def _linearize_functional_common(f, x, get_sigma_points):
    x = get_mvnsqrt(x)
    m_x, chol_x = x

    x_pts = get_sigma_points(x)

    f_pts = jax.vmap(f)(x_pts.points)
    m_f = jnp.dot(x_pts.wm, f_pts)

    Psi_x = _cov(x_pts.wc, x_pts.points, m_x, f_pts, m_f)
    F_x = cho_solve((chol_x, True), Psi_x).T

    return F_x, x_pts, f_pts, m_f


def _concatenate_mvns(x, q):
    # This code implicitly assumes that X and Q are independent multivariate Gaussians.
    m_x, chol_x = x
    m_q, chol_q = q
    m_xq = jnp.concatenate([m_x, m_q])
    chol_xq = block_diag(chol_x, chol_q)
    return MVNSqrt(m_xq, chol_xq)

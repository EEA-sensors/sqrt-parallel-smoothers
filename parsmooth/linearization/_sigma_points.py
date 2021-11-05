from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_solve

from parsmooth._base import MVNSqrt, FunctionalModel, ConditionalMomentsModel
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


def linearize_conditional(model, x, get_sigma_points):
    first_and_second_moments = model.first_and_second_moments
    F_x, x_pts, f_pts, m_f, C_f_pts = linearize_conditional_common(first_and_second_moments, x,
                                                                   get_sigma_points)
    if isinstance(x, MVNSqrt):
        m_x, chol_x = x

        sqrt_Phi = jnp.sqrt(x_pts.wc[:, None]) * (f_pts - m_f[None, :])
        sqrt_Phi = tria(sqrt_Phi.T)

        chol_L = cholesky_update_many(sqrt_Phi, (F_x @ chol_x).T, -1.)
        chol_f = jnp.sum(x_pts.wm[:, None, None] * C_f_pts, 0)

        L = tria(jnp.concatenate([chol_L, chol_f], axis=1))
        return F_x, L, m_f - F_x @ m_x

    m_x, cov_x = x
    Phi = _cov(x_pts.wc, f_pts, m_f, f_pts, m_f)
    cov_f = jnp.sum(x_pts.wm[:, None, None] * C_f_pts, 0)
    L = Phi - F_x @ cov_x @ F_x.T + cov_f
    return F_x, L, m_f - F_x @ m_x


def linearize_conditional_common(first_and_second_moments, x, get_sigma_points):
    x = get_mvnsqrt(x)
    m_x, chol_x = x
    x_pts = get_sigma_points(x)

    f_pts, C_pts = jax.vmap(first_and_second_moments)(x_pts.points)
    m_f = jnp.dot(x_pts.wm, f_pts)

    Psi_x = _cov(x_pts.wc, x_pts.points, m_x, f_pts, m_f)
    F_x = cho_solve((chol_x, True), Psi_x).T
    return F_x, x_pts, f_pts, m_f, C_pts


def linearize_functional(model: FunctionalModel, x, get_sigma_points):
    f, q, is_additive = model
    if is_additive:
        m_q, chol_or_cov_q = q

        def first_and_second_order(y):
            m_f = f(y) + m_q
            return m_f, chol_or_cov_q

        conditional_moment_model = ConditionalMomentsModel(first_and_second_order)
    else:
        conditional_moment_model = _get_conditional_moment_model_from_non_additive(f, q, get_sigma_points)

    return linearize_conditional(conditional_moment_model, x, get_sigma_points)


def _get_conditional_moment_model_from_non_additive(f, q, get_sigma_points):
    def mean_and_cov(x):
        F_q, wc, f_pts, m_f = _linearize_functional_common(lambda q_: f(x, q_), q, get_sigma_points)
        cov_f = _cov(wc, f_pts, m_f, f_pts, m_f)
        return m_f, cov_f

    def mean_and_chol(x):
        F_q, wc, f_pts, m_f = _linearize_functional_common(lambda q_: f(x, q_), q, get_sigma_points)
        sqrt_Phi = jnp.sqrt(wc[:, None]) * (f_pts - m_f[None, :])
        sqrt_Phi = tria(sqrt_Phi.T)
        return m_f, sqrt_Phi

    if isinstance(q, MVNSqrt):
        return ConditionalMomentsModel(mean_and_chol)
    return ConditionalMomentsModel(mean_and_cov)


def _linearize_functional_common(f, x, get_sigma_points):
    x = get_mvnsqrt(x)
    m_x, chol_x = x
    x_pts = get_sigma_points(x)

    f_pts = jax.vmap(f)(x_pts.points)
    m_f = jnp.dot(x_pts.wm, f_pts)

    Psi_x = _cov(x_pts.wc, x_pts.points, m_x, f_pts, m_f)

    F_x = cho_solve((chol_x, True), Psi_x).T
    return F_x, x_pts.wc, f_pts, m_f

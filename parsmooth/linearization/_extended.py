from typing import Union

import jax
import jax.numpy as jnp

from parsmooth._base import FunctionalModel, ConditionalMomentsModel, MVNSqrt, are_inputs_compatible, MVNStandard


def linearize(model: Union[FunctionalModel, ConditionalMomentsModel], x: Union[MVNSqrt, MVNStandard]):
    """
    Extended linearization for a non-linear function f(x, q). If the function is linear, JAX Jacobian calculation will
    simply return the matrices without additional complexity.

    Parameters
    ----------
    model: Union[FunctionalModel, ConditionalMomentsModel]
        The function to be called on x and q
    x: Union[MVNSqrt, MVNStandard]
        x-coordinate state at which to linearize f

    Returns
    -------
    F_x, F_q, res: jnp.ndarray
        The linearization parameters
    chol_q or cov_q: jnp.ndarray
        Either the cholesky or the full-rank modified covariance matrix.
    """
    if isinstance(model, FunctionalModel):
        f, q, is_additive = model
        are_inputs_compatible(x, q)

        m_x, _ = x
        if isinstance(x, MVNSqrt):
            return _sqrt_linearize_callable(f, m_x, *q, is_additive)
        return _standard_linearize_callable(f, m_x, *q, is_additive)

    else:
        first_and_second_moments = model.first_and_second_moments
        if isinstance(x, MVNSqrt):
            return _sqrt_linearize_conditional(first_and_second_moments, x)
        return _standard_linearize_conditional(first_and_second_moments, x)


def _standard_linearize_conditional(mean_and_cov, x):
    m, p = x

    # FIXME: When value_and_jac arrives...
    c_m_val, c_cov_cal, = mean_and_cov(m)
    F, _ = jax.jacfwd(mean_and_cov)(m)

    m, p = x
    b = c_m_val - F @ m
    return F, c_cov_cal, b


def _sqrt_linearize_conditional(mean_and_chol, x):
    m, _ = x

    # FIXME: When value_and_jac arrives...
    c_m_val, c_chol_val = mean_and_chol(m)
    F, _ = jax.jacfwd(mean_and_chol)(m)

    b = c_m_val - F @ m
    return F, c_chol_val, b


def _linearize_callable_common(f, x, q, is_additive):
    dim_x = x.shape[0]
    dim_q = q.shape[0]
    if not is_additive:
        f_val = f(x, q)
        if dim_q > dim_x:
            return f_val, *jax.jacrev(f, (0, 1))(x, q)
        return f_val, *jax.jacfwd(f, (0, 1))(x, q)
    return f(x) + q, jax.jacfwd(f)(x)


def _standard_linearize_callable(f, x, m_q, cov_q, is_additive):
    if is_additive:
        res, F_x = _linearize_callable_common(f, x, m_q, is_additive)
        return F_x, cov_q, res - F_x @ x
    res, F_x, F_q = _linearize_callable_common(f, x, m_q, is_additive)  # noqa
    return F_x, F_q @ cov_q @ F_q.T, res - F_x @ x


def _sqrt_linearize_callable(f, x, m_q, chol_q, is_additive):
    if is_additive:
        res, F_x = _linearize_callable_common(f, x, m_q, is_additive)
        return F_x, chol_q, res - F_x @ x
    res, F_x, F_q = _linearize_callable_common(f, x, m_q, is_additive)  # noqa
    return F_x, F_q @ chol_q, res - F_x @ x

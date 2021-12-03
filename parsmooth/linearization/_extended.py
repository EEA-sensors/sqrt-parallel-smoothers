from typing import Any, Tuple, Union

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
        f, q = model
        are_inputs_compatible(x, q)

        m_x, _ = x
        if isinstance(x, MVNSqrt):
            return _sqrt_linearize_callable(f, m_x, *q)
        return _standard_linearize_callable(f, m_x, *q)

    else:
        if isinstance(x, MVNSqrt):
            return _sqrt_linearize_conditional(model.conditional_mean, model.conditional_covariance_or_cholesky, x)
        return _standard_linearize_conditional(model.conditional_mean, model.conditional_covariance_or_cholesky, x)


def _standard_linearize_conditional(c_m, c_cov, x):
    m, p = x
    F = jax.jacfwd(c_m, 0)(m)
    b = c_m(m) - F @ m
    Cov = c_cov(m)
    return F, Cov, b


def _sqrt_linearize_conditional(c_m, c_chol, x):
    m, _ = x
    F = jax.jacfwd(c_m, 0)(m)
    b = c_m(m) - F @ m
    Chol = c_chol(m)
    return F, Chol, b


def _linearize_callable_common(f, x) -> Tuple[Any, Any]:
    return f(x), jax.jacfwd(f, 0)(x)


def _standard_linearize_callable(f, x, m_q, cov_q):
    res, F_x = _linearize_callable_common(f, x)
    return F_x, cov_q, res - F_x @ x + m_q


def _sqrt_linearize_callable(f, x, m_q, chol_q):
    res, F_x = _linearize_callable_common(f, x)
    return F_x, chol_q, res - F_x @ x + m_q

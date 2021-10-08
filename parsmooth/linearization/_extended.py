from typing import Any, Tuple

import jax
import jax.numpy as jnp

from parsmooth._base import FunctionalModel, ConditionalMomentsModel, MVNSqrt, are_inputs_compatible


def linearize(model, x):
    """
    Extended linearization for a non-linear function f(x, q). If the function is linear, JAX Jacobian calculation will
    simply return the matrices without additional complexity.

    Parameters
    ----------
    model: FunctionalModel or ConditionalMomentsModel
        The function to be called on x and q
    x: MVNStandard or MVNSqrt
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

    if isinstance(model, ConditionalMomentsModel):
        if isinstance(x, MVNSqrt):
            return _sqrt_linearize_conditional(model.conditional_mean, model.conditional_covariance_or_cholesky, x)
        return _standard_linearize_conditional(model.conditional_mean, model.conditional_covariance_or_cholesky, x)


def _linearize_conditional_common(c_m, c_cov, x) -> Tuple[Any, Any, Any]:
    m, p = x
    d_cm = jax.jacfwd(c_m, 0)
    return c_m(m), c_cov(m) + d_cm(m) @ p @ d_cm(m).T, d_cm(m) @ p


def _standard_linearize_conditional(c_m, c_cov, x):
    E_y, Cov_y, Cov_y_x = _linearize_conditional_common(c_m, c_cov, x)
    m, p = x
    F = jnp.linalg.solve(p.T, Cov_y_x.T).T
    b = E_y - F @ m
    Cov = Cov_y - F @ p @ F.T
    return F, Cov, b


def _sqrt_linearize_conditional(c_m, c_chol, x):
    m, _ = x
    F = jax.jacfwd(c_m, 0)(m)
    b = c_m(m) - F @ m
    Chol = c_chol(m)
    return F, Chol, b


def _linearize_callable_common(f, x, q) -> Tuple[Any, Any, Any]:
    dim_x = x.shape[0]
    dim_q = q.shape[0]
    if dim_q > dim_x:
        return f(x, q), *jax.jacrev(f, (0, 1))(x, q)  # noqa: this really is a 3-tuple.
    return f(x, q), *jax.jacfwd(f, (0, 1))(x, q)  # noqa: this really is a 3-tuple.


def _standard_linearize_callable(f, x, q, Q):
    res, F_x, F_q = _linearize_callable_common(f, x, q)
    return F_x, F_q @ Q @ F_q.T, res - F_x @ x


def _sqrt_linearize_callable(f, x, q, cholQ):
    res, F_x, F_q = _linearize_callable_common(f, x, q)
    return F_x, F_q @ cholQ, res - F_x @ x

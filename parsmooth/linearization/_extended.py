from typing import Any, Tuple

import jax
import jax.numpy as jnp

from parsmooth._base import FunctionalModel, ConditionalMomentsModel, MVNSqrt, are_inputs_compatible


def linearize(f, x):
    """
    Extended linearization for a non-linear function f(x, q). If the function is linear, JAX Jacobian calculation will
    simply return the matrices without additional complexity.

    Parameters
    ----------
    f: FunctionalModel or ConditionalMomentsModel
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
    if isinstance(f, FunctionalModel):
        f, q = f
        are_inputs_compatible(x, q)

        m_x, _ = x
        if isinstance(x, MVNSqrt):
            return _sqrt_linearize_callable(f, m_x, *q)
        return _standard_linearize_callable(f, m_x, *q)

    if isinstance(f, ConditionalMomentsModel):
        if isinstance(x, MVNSqrt):
            return _sqrt_conditional_linearize(f, x)
        return _standard_conditional_linearize(f, x)


def _conditional_linearize_common(f, x) -> Tuple[Any, Any, Any]:
    E_f, V_f = f
    E_x, V_x = x
    dE_f = jax.jacfwd(E_f, 0)
    return E_f(E_x), V_f(E_x) + dE_f(E_x) @ V_x @ dE_f(E_x).T, dE_f(E_x) @ V_x


def _standard_conditional_linearize(f, x):
    E_y, V_y, Cov_y_x = _conditional_linearize_common(f, x)
    E_x, V_x = x
    C = jnp.linalg.solve(V_x.T, Cov_y_x.T).T
    d = E_y - C @ E_x
    COV = V_y - C @ V_x @ C.T
    return C, COV, d


def _sqrt_conditional_linearize(f, x):
    E_f, Chol_f = f
    E_x, chol_x = x
    C = jax.jacfwd(E_f, 0)(E_x)
    d = E_f(E_x) - C @ E_x
    Chol = Chol_f(E_x)
    return C, Chol, d


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

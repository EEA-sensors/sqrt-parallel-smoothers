from typing import Any, Tuple

import jax

from parsmooth._base import FunctionalModel, MVNSqrt, are_inputs_compatible


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

    raise NotImplementedError("Not implemented yet")


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

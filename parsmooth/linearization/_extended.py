import jax

from parsmooth.linearization._common import fix_mvn


def linearize(f, x, q, sqrt=False):
    """
    Extended linearization for a non-linear function f(x, q). If the function is linear, JAX Jacobian calculation will
    simply return the matrices without additional complexity.

    Parameters
    ----------
    f: Callable
        The function to be called on x and q
    x: MVNParams
        x-coordinate state at which to linearize f
    q: MVNParams
        q-coordinate state at which to linearize f
    sqrt: bool, optional
        return the sqrt of the modified noise covariance. Default is False

    Returns
    -------
    F_x, F_q, res: jnp.ndarray
        The linearization parameters
    chol_q or cov_q: jnp.ndarray
        Either the cholesky or the full-rank modified covariance matrix.
    """
    if callable(f):
        m_x, *_ = x
        q = fix_mvn(q)
        m_q, cov_q, chol_q = q
        if not sqrt:
            return _linearize_callable(f, m_x, m_q) + (chol_q,)

        raise NotImplementedError("Not implemented yet")
    raise NotImplementedError("Not implemented yet")


def _linearize_callable(f, x, q):
    dim_x = x.shape[0]
    dim_q = q.shape[0]

    if dim_q > dim_x:
        F_x, F_q = jax.jacrev(f, (0, 1))(x, q)
    else:
        F_x, F_q = jax.jacfwd(f, (0, 1))(x, q)

    res = f(x, q)

    return F_x, F_q, res

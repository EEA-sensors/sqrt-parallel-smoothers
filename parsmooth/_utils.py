from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
from jax import custom_vjp, vjp
from jax.custom_derivatives import closure_convert
from jax.flatten_util import ravel_pytree
from jax.lax import while_loop


def cholesky_update_many(chol_init, update_vectors, multiplier):
    def body(chol, update_vector):
        res = _cholesky_update(chol, update_vector, multiplier=multiplier)
        return res, None

    final_chol, _ = jax.lax.scan(body, chol_init, update_vectors)
    return final_chol


def tria(A):
    return qr(A.T).T


def _set_diagonal(x, y):
    N, _ = x.shape
    i, j = jnp.diag_indices(N)
    return x.at[i, j].set(y)


def _set_triu(x, val):
    N, _ = x.shape
    i = jnp.triu_indices(N, 1)
    return x.at[i].set(val)


def _cholesky_update(chol, update_vector, multiplier=1.):
    chol_diag = jnp.diag(chol)

    # The algorithm in [1] is implemented as a double for loop. We can treat
    # the inner loop in Algorithm 3.1 as a vector operation, and thus the
    # whole algorithm as a single for loop, and hence can use a `tf.scan`
    # on it.

    # We use for accumulation omega and b as defined in Algorithm 3.1, since
    # these are updated per iteration.

    def scan_body(carry, inp):
        _, _, omega, b = carry
        index, diagonal_member, col = inp
        omega_at_index = omega[..., index]

        # Line 4
        new_diagonal_member = jnp.sqrt(jnp.square(diagonal_member) +
                                       multiplier / b * jnp.square(omega_at_index))
        # `scaling_factor` is the same as `gamma` on Line 5.
        scaling_factor = (jnp.square(diagonal_member) * b + multiplier * jnp.square(omega_at_index))

        # The following updates are the same as the for loop in lines 6-8.
        omega = omega - (omega_at_index / diagonal_member)[..., None] * col
        new_col = new_diagonal_member[..., None] * (
                col / diagonal_member[..., None] +
                (multiplier * omega_at_index / scaling_factor)[..., None] * omega)
        b = b + multiplier * jnp.square(omega_at_index / diagonal_member)
        return (new_diagonal_member, new_col, omega, b), (new_diagonal_member, new_col, omega, b)

    # We will scan over the columns.
    chol = chol.T

    _, (new_diag, new_chol, _, _) = jax.lax.scan(scan_body,
                                                 (0., jnp.zeros_like(chol[0]), update_vector, 1.),
                                                 (jnp.arange(0, chol.shape[0]), chol_diag, chol),
                                                 )

    new_chol = new_chol.T
    new_chol = _set_diagonal(new_chol, new_diag)
    new_chol = _set_triu(new_chol, 0.)
    new_chol = jnp.where(jnp.isfinite(new_chol), new_chol, 0.)
    return new_chol


def none_or_shift(x, shift):
    if x is None:
        return None
    if shift > 0:
        return jax.tree_map(lambda z: z[shift:], x)
    return jax.tree_map(lambda z: z[:shift], x)


def none_or_concat(x, y, position=1):
    if x is None or y is None:
        return None
    if position == 1:
        return jax.tree_map(lambda a, b: jnp.concatenate([a[None, ...], b]), y, x)
    else:
        return jax.tree_map(lambda a, b: jnp.concatenate([b, a[None, ...]]), y, x)


# FIXED POINT UTIL

def fixed_point(f, x0, criterion):
    converted_fn, aux_args = closure_convert(f, x0)
    return _fixed_point(converted_fn, aux_args, x0, criterion)


@partial(custom_vjp, nondiff_argnums=(0, 3))
def _fixed_point(f, params, x0, criterion):
    return __fixed_point(f, params, x0, criterion)[0]


def _fixed_point_fwd(f, params, x0, criterion):
    x_star, n_iter = __fixed_point(f, params, x0, criterion)
    return x_star, (params, x_star, n_iter)


def _fixed_point_rev(f, _criterion, res, x_star_bar):
    params, x_star, n_iter = res
    _, vjp_theta = vjp(lambda p: f(x_star, *p), params)
    theta_bar, = vjp_theta(__fixed_point(partial(_rev_iter, f),
                                         (params, x_star, x_star_bar),
                                         x_star_bar,
                                         lambda i, *_: i < n_iter + 1)[0])
    return theta_bar, jax.tree_map(jnp.zeros_like, x_star)


def _rev_iter(f, u, *packed):
    params, x_star, x_star_bar = packed
    _, vjp_x = vjp(lambda x: f(x, *params), x_star)
    ravelled_x_star_bar, unravel_fn = ravel_pytree(x_star_bar)
    ravelled_vjp_x_u, _ = ravel_pytree(vjp_x(u)[0])
    return unravel_fn(ravelled_x_star_bar + ravelled_vjp_x_u)


def __fixed_point(f, params, x0, criterion):
    def cond_fun(carry):
        i, x_prev, x = carry
        return criterion(i, x_prev, x)

    def body_fun(carry):
        i, _, x = carry
        return i + 1, x, f(x, *params)

    n_iter, _, x_star = while_loop(cond_fun, body_fun, (1, x0, f(x0, *params)))
    return x_star, n_iter


_fixed_point.defvjp(_fixed_point_fwd, _fixed_point_rev)


def mvn_loglikelihood(x, chol_cov):
    """multivariate normal"""
    dim = chol_cov.shape[0]
    y = jlinalg.solve_triangular(chol_cov, x, lower=True)
    normalizing_constant = (
            jnp.sum(jnp.log(jnp.abs(jnp.diag(chol_cov)))) + dim * jnp.log(2 * jnp.pi) / 2.0
    )
    norm_y = jnp.sum(y * y, -1)
    return -0.5 * norm_y - normalizing_constant


@jax.custom_jvp
def qr(A: jnp.ndarray):
    """The JAX provided implementation is not parallelizable using VMAP. As a consequence, we have to rewrite it..."""
    return _qr(A)


# @partial(jax.jit, static_argnums=(1,))
def _qr(A: jnp.ndarray, return_q=False):
    m, n = A.shape
    min_ = min(m, n)
    if return_q:
        Q = jnp.eye(m)

    for j in range(min_):
        # Apply Householder transformation.
        v, tau = _householder(A[j:, j])

        H = jnp.eye(m)
        H = H.at[j:, j:].add(-tau * (v[:, None] @ v[None, :]))

        A = H @ A
        if return_q:
            Q = H @ Q  # noqa

    R = jnp.triu(A[:min_, :min_])
    if return_q:
        return Q[:n].T, R  # noqa
    else:
        return R


def _householder(a):
    if a.dtype == jnp.float64:
        eps = 1e-9
    else:
        eps = 1e-7

    alpha = a[0]
    s = jnp.sum(a[1:] ** 2)
    cond = s < eps

    def if_not_cond(v):
        t = (alpha ** 2 + s) ** 0.5
        v0 = jax.lax.cond(alpha <= 0, lambda _: alpha - t, lambda _: -s / (alpha + t), None)
        tau = 2 * v0 ** 2 / (s + v0 ** 2)
        v = v / v0
        v = v.at[0].set(1.)
        return v, tau

    return jax.lax.cond(cond, lambda v: (v, 0.), if_not_cond, a)


def qr_jvp_rule(primals, tangents):
    x, = primals
    dx, = tangents
    q, r = _qr(x, True)
    m, n = x.shape
    min_ = min(m, n)
    if m < n:
        dx = dx[:, :m]
    dx_rinv = jax.lax.linalg.triangular_solve(r, dx)
    qt_dx_rinv = jnp.matmul(q.T, dx_rinv)
    qt_dx_rinv_lower = jnp.tril(qt_dx_rinv, -1)
    do = qt_dx_rinv_lower - qt_dx_rinv_lower.T  # This is skew-symmetric
    # The following correction is necessary for complex inputs
    do = do + jnp.eye(min_, dtype=do.dtype) * (qt_dx_rinv - jnp.real(qt_dx_rinv))
    dr = jnp.matmul(qt_dx_rinv - do, r)
    return r, dr


qr.defjvp(qr_jvp_rule)

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg


def cholesky_update_many(chol_init, update_vectors, multiplier):
    def body(chol, update_vector):
        res = _cholesky_update(chol, update_vector, multiplier=multiplier)
        return res, None

    final_chol, _ = jax.lax.scan(body, chol_init, update_vectors)
    return final_chol


def tria(A):
    tria_A = jlinalg.qr(A.T, mode='economic')[1].T
    return tria_A


def _set_diagonal(x, y):
    N, _ = x.shape
    i, j = jnp.diag_indices(N)
    return x.at[i, j].set(y)


def _set_triu(x, val):
    N, _ = x.shape
    i = jnp.triu_indices(N, 1)
    return x.at[i].set(val)


def _cholesky_update(chol, update_vector, multiplier=1.):
    # FIXME: This is fixed by the PR https://github.com/tensorflow/probability/pull/1423/files.
    #        Will be deleted once it is fixed by Tensorflow Probability

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
        new_diagonal_member = jnp.sqrt(jnp.abs(jnp.square(diagonal_member) +
                                               multiplier / b * jnp.square(omega_at_index)))
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
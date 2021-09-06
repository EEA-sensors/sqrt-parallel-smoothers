import jax
import jax.scipy.linalg as jlinalg
import tensorflow_probability.substrates.jax as tfp


def cholesky_update_many(chol_init, update_vectors, multiplier):
    def body(chol, update_vector):
        res = tfp.math.cholesky_update(chol, update_vector, multiplier=multiplier)
        return res, None

    final_chol, _ = jax.lax.scan(body, chol_init, update_vectors)
    return final_chol


def tria(A):
    tria_A = jlinalg.qr(A.T, mode='economic')[1].T
    return tria_A

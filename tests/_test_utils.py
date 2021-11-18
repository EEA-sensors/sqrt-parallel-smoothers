import jax.numpy as jnp
import numpy as np

from parsmooth._base import MVNStandard, MVNSqrt


def get_system(dim_x, dim_y, dtype=np.float32):
    m = jnp.array(np.random.randn(dim_x), dtype)
    cholP = jnp.array(np.random.rand(dim_x, dim_x), dtype)
    cholP = cholP.at[np.triu_indices(dim_x, 1)].set(0.)
    P = cholP @ cholP.T

    cholR = jnp.array(np.random.rand(dim_y, dim_y), dtype)
    cholR = cholR.at[np.triu_indices(dim_y, 1)].set(0.)
    R = cholR @ cholR.T

    H = jnp.eye(dim_y, dim_x, dtype=dtype)
    c = jnp.array(np.random.randn(dim_y), dtype)
    y = jnp.array(np.random.randn(dim_y), dtype)

    chol_x = MVNSqrt(m, cholP)
    x = MVNStandard(m, P)
    return x, chol_x, H, R, cholR, c, y

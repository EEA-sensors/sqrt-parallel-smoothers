from jax import numpy as jnp

from parsmooth._base import MVNParams


def fix_mvn(x):
    m_x, cov_x, chol_x = x

    if chol_x is None:
        chol_x = jnp.linalg.cholesky(cov_x)
    if cov_x is None:
        cov_x = chol_x @ chol_x.T
    return MVNParams(m_x, cov_x, chol_x)

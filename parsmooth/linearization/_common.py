from typing import Union

from jax import numpy as jnp

from parsmooth._base import MVNStandard, MVNSqrt


def get_mvnsqrt(x: Union[MVNSqrt, MVNStandard]):
    if isinstance(x, MVNSqrt):
        return x
    m_x, cov_x = x
    chol_x = jnp.linalg.cholesky(cov_x)
    return MVNSqrt(m_x, chol_x)

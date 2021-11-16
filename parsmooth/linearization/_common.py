from typing import Union

from jax import numpy as jnp

from parsmooth._base import MVNStandard, MVNSqrt, FunctionalModel, ConditionalMomentsModel
from parsmooth._utils import tria


def get_mvnsqrt(x: Union[MVNSqrt, MVNStandard]):
    if isinstance(x, MVNSqrt):
        return x
    m_x, cov_x = x
    chol_x = jnp.linalg.cholesky(cov_x)
    return MVNSqrt(m_x, chol_x)


def get_conditional_model(f, q: Union[MVNSqrt, MVNStandard], linearization_method):
    """

    Parameters
    ----------
    f: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        A general functional model with a non-linearity in q too.
    q: Union[MVNSqrt, MVNStandard]
        The noise Gaussian
    linearization_method:
        The linearization used for the `q` part.

    Returns
    -------
    conditional_model: ConditionalMomentsModel
        The resulting conditional moments model that can be used in lieu of the dounbly non-linear function `f`

    """
    # FIXME: this has twice the amount of logic we need. We have to ask users to pass ONE function that returns
    #        both the conditional mean and conditional covariance at the same time. This will avoid recomputing the same
    #        thing twice.

    # FIXME: This will not work if dim(q) != dim(x) yet. We need to rewrite the conditional moments linearization
    #        a bit to make it work for free.

    sqrt = isinstance(q, MVNSqrt)
    try:
        f(q.mean, q.mean)
    except:  # noqa
        raise NotImplementedError("`x` and `q` with different dimensions are not supported yet.")

    if sqrt:
        additional_noise = MVNSqrt(jnp.zeros_like(q.mean), jnp.zeros_like(q.chol))
    else:
        additional_noise = MVNStandard(jnp.zeros_like(q.mean), jnp.zeros_like(q.cov))

    def mean(x):
        model = FunctionalModel(lambda q_: f(x, q_), additional_noise)
        F, _, bias = linearization_method(model, q)
        conditional_mean = F @ q.mean + bias
        return conditional_mean

    def chol_or_cov(x):
        model = FunctionalModel(lambda q_: f(x, q_), additional_noise)
        F, cov_or_chol_val, _ = linearization_method(model, q)
        if sqrt:
            return tria(jnp.concatenate([F @ q.chol, cov_or_chol_val], axis=1))
        return F @ q.cov @ F.T + cov_or_chol_val

    return ConditionalMomentsModel(mean, chol_or_cov)

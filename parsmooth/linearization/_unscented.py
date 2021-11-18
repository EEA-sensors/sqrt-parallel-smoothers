from typing import Tuple, Union, Optional

import jax.numpy as jnp
import jax.ops
import numpy as np

from parsmooth._base import MVNStandard, FunctionalModel, ConditionalMomentsModel, MVNSqrt
from parsmooth.linearization._sigma_points import SigmaPoints, linearize_functional, linearize_conditional


def linearize(model: Union[FunctionalModel, ConditionalMomentsModel],
              x: Union[MVNSqrt, MVNStandard],
              alpha: float = 1., beta: float = 0., kappa: float = None):
    """
    Unscented linearization for a non-linear function f(x, q). While this may look inefficient for functions with
    additive noise, JAX relies on XLA which compresses linear operations. This means that in practice our code will only
    slow down tracing (compilation) time and not run time.

    Parameters
    ----------
    model: Union[FunctionalModel, ConditionalMomentsModel]
        The function to be called on x and q
    x: Union[MVNSqrt, MVNStandard]
        x-coordinate state at which to linearize f
    alpha, beta, kappa: float, optional
        Parameters of the unscented transform. Default is `alpha=1.`, `beta=0.` and `kappa=3-n`

    Returns
    -------
    F_x, F_q, res: jnp.ndarray
        The linearization parameters
    chol_q or cov_q: jnp.ndarray
        Either the cholesky or the full-rank modified covariance matrix.
    """
    get_sigma_points = lambda mvn: _get_sigma_points(mvn, alpha, beta, kappa)
    if isinstance(model, FunctionalModel):
        f, q = model
        return linearize_functional(f, x, q, get_sigma_points)
    conditional_mean, conditional_covariance_or_cholesky = model
    return linearize_conditional(conditional_mean, conditional_covariance_or_cholesky, x, get_sigma_points)


def _get_sigma_points(
        mvn: MVNSqrt, alpha: float, beta: float, kappa: Optional[float]
) -> Tuple[SigmaPoints, jnp.ndarray]:
    """ Computes the sigma-points for a given mv normal distribution
    The number of sigma-points is 2*n_dim

    Parameters
    ----------
    mvn: MVNSqrt
        Mean and Sqrt covariance of the distribution
    alpha, beta, kappa: float, optional
        Parameters of the unscented transform. Default is `alpha=0.5`, `beta=2.` and `kappa=3-n`

    Returns
    -------
    out: SigmaPoints
        sigma points for the spherical cubature transform
    xi: jnp.ndarray
        Unit sigma points vectors used
    """
    mean, chol = mvn
    n_dim = mean.shape[0]
    dtype = mean.dtype
    if kappa is None:
        kappa = 3. + n_dim
    wm, wc, lamda = _unscented_weights(n_dim, alpha, beta, kappa, dtype)
    scaled_chol = jnp.sqrt(n_dim + lamda) * mvn.chol

    zeros = jnp.zeros_like(scaled_chol, shape=(1, n_dim))
    sigma_points = mean[None, :] + jnp.concatenate([zeros, scaled_chol.T, -scaled_chol.T], axis=0)
    return SigmaPoints(sigma_points, wm, wc)


def _unscented_weights(n_dim: int, alpha: float, beta: float, kappa: Optional[float], dtype) -> Tuple[
    np.ndarray, np.ndarray, float]:
    """ Computes the weights associated with the spherical cubature method.
    The number of sigma-points is 2 * n_dim

    Parameters
    ----------
    n_dim: int
        Dimension of the space
    alpha, beta, kappa: float, optional
        Parameters of the unscented transform. Default is `alpha=0.5`, `beta=2.` and `kappa=3-n`
    dtype:
        dtype of the output
    Returns
    -------
    wm: np.ndarray
        Weights means
    wc: np.ndarray
        Weights covariances
    lamda: float
        Correction for the covariance of the sigma-points
    """

    lamda = alpha ** 2 * (n_dim + kappa) - n_dim
    wm = jnp.full(2 * n_dim + 1, 1 / (2 * (n_dim + lamda)), dtype=dtype)

    wm = jax.ops.index_update(wm, 0, lamda / (n_dim + lamda), indices_are_sorted=True, unique_indices=True)
    wc = jax.ops.index_update(wm, 0, lamda / (n_dim + lamda) + (1 - alpha ** 2 + beta), indices_are_sorted=True,
                              unique_indices=True)
    return wm, wc, lamda

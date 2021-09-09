from typing import Tuple

import jax.numpy as jnp
import numpy as np

from parsmooth._base import MVNParams, FunctionalModel, ConditionalMomentsModel
from parsmooth.linearization._sigma_points import SigmaPoints, linearize_callable


def linearize(f, x, sqrt=False):
    """
    Cubature linearization for a non-linear function f(x, q). While this may look inefficient for functions with
    additive noise, JAX relies on XLA which compresses linear operations. This means that in practice our code will only
    slow down tracing (compilation) time and not run time.

    Parameters
    ----------
    f: FunctionalModel or ConditionalMomentsModel
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
    if isinstance(f, FunctionalModel):
        f, q = f
        return linearize_callable(f, x, q, _get_sigma_points, sqrt)
    raise NotImplementedError("Not implemented yet")


def _get_sigma_points(mvn: MVNParams) -> Tuple[SigmaPoints, jnp.ndarray]:
    """ Computes the sigma-points for a given mv normal distribution
    The number of sigma-points is 2*n_dim

    Parameters
    ----------
    mvn: MVNParams
        Mean and Covariance of the distribution

    Returns
    -------
    out: SigmaPoints
        sigma points for the spherical cubature transform
    xi: jnp.ndarray
        Unit sigma points vectors used
    """
    mean = mvn.mean
    chol = mvn.chol
    n_dim = mean.shape[0]

    wm, wc, xi = _cubature_weights(n_dim)

    sigma_points = jnp.repeat(mean.reshape(1, -1), wm.shape[0], axis=0) + jnp.dot(chol, xi.T).T

    return SigmaPoints(sigma_points, wm, wc)


def _cubature_weights(n_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Computes the weights associated with the spherical cubature method.
    The number of sigma-points is 2 * n_dim

    Parameters
    ----------
    n_dim: int
        Dimensionality of the problem

    Returns
    -------
    wm: np.ndarray
        Weights means
    wc: np.ndarray
        Weights covariances
    xi: np.ndarray
        Orthogonal vectors
    """
    wm = np.ones(shape=(2 * n_dim,)) / (2 * n_dim)
    wc = wm
    xi = np.concatenate([np.eye(n_dim), -np.eye(n_dim)], axis=0) * np.sqrt(n_dim)

    return wm, wc, xi
